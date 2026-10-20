# Copyright (C) 2026 Open Navigation LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only tests for SAM3 node callback and shutdown serialization."""

import importlib
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def node_module(monkeypatch):
    """Import the node while permitting an unbuilt local message package."""
    pytest.importorskip('torch')
    pytest.importorskip('rclpy')
    try:
        importlib.import_module('opennav_sam3_msgs.srv')
    except ImportError:
        package = ModuleType('opennav_sam3_msgs')
        package.__path__ = []
        service = ModuleType('opennav_sam3_msgs.srv')
        service.ChangePrompt = object
        monkeypatch.setitem(sys.modules, 'opennav_sam3_msgs', package)
        monkeypatch.setitem(sys.modules, 'opennav_sam3_msgs.srv', service)
    return importlib.import_module('opennav_sam3_inference.sam3_node')


class _Logger:
    """Discard log messages from uninitialized node test doubles."""

    def info(self, _message):
        """Discard an info message."""

    def warn(self, _message):
        """Discard a warning message."""

    def error(self, _message):
        """Discard an error message."""


class _BlockingContext:
    """Expose when a callback is waiting to enter its critical section."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.held = False

    def __enter__(self):
        self.entered.set()
        if not self.release.wait(timeout=2):
            raise TimeoutError('test did not release callback lock')
        self.held = True
        return self

    def __exit__(self, *_args):
        self.held = False


def _prompt_node(module, lock):
    """Create the state needed by the prompt service callback."""
    reset_calls = []
    published = []

    def reset_prompts(prompts):
        assert lock.held
        reset_calls.append(list(prompts))

    node = SimpleNamespace(
        _infer_lock=lock,
        _runtime_closed=False,
        _prompts=['floor'],
        _class_ids=[1],
        _prompt_to_class_id={'floor': 1},
        _live=SimpleNamespace(reset_prompts=reset_prompts),
        _validate_prompt_config=module.Sam3InferenceNode._validate_prompt_config,
        _publish_label_info=lambda: published.append(True),
        get_logger=lambda: _Logger(),
    )
    return node, reset_calls, published


def test_prompt_change_waits_for_inference_lock(node_module):
    """Do not mutate a live inference session from a concurrent service."""
    lock = _BlockingContext()
    node, reset_calls, published = _prompt_node(node_module, lock)
    request = SimpleNamespace(prompts=['wall'], class_ids=[2])
    response = SimpleNamespace(success=None, message='')
    finished = threading.Event()

    def invoke():
        node_module.Sam3InferenceNode._on_change_prompt(
            node,
            request,
            response,
        )
        finished.set()

    worker = threading.Thread(target=invoke)
    worker.start()
    assert lock.entered.wait(timeout=1)
    assert not reset_calls
    assert not finished.is_set()

    lock.release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert reset_calls == [['wall']]
    assert published == [True]
    assert node._prompt_to_class_id == {'wall': 2}
    assert response.success is True


def test_enable_change_uses_inference_lock(node_module):
    """Finish an in-flight callback before acknowledging disable."""
    lock = _BlockingContext()
    node = SimpleNamespace(
        _infer_lock=lock,
        _runtime_closed=False,
        _enabled=True,
        get_logger=lambda: _Logger(),
    )
    request = SimpleNamespace(data=False)
    response = SimpleNamespace(success=None, message='')
    finished = threading.Event()

    def invoke():
        node_module.Sam3InferenceNode._on_enable(node, request, response)
        finished.set()

    worker = threading.Thread(target=invoke)
    worker.start()
    assert lock.entered.wait(timeout=1)
    assert node._enabled is True
    assert not finished.is_set()

    lock.release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert node._enabled is False
    assert response.success is True


def test_image_rechecks_enabled_state_after_lock_acquisition(node_module):
    """Drop a frame when disable wins the race before lock acquisition."""
    released = []
    node = SimpleNamespace(
        _enabled=True,
        _runtime_closed=False,
        get_logger=lambda: _Logger(),
    )

    class DisableBeforeAcquire:
        """Simulate disable completing between the two enabled checks."""

        def acquire(self, *, blocking):
            assert blocking is False
            node._enabled = False
            return True

        def release(self):
            released.append(True)

    node._infer_lock = DisableBeforeAcquire()
    node._bridge = SimpleNamespace(
        imgmsg_to_cv2=lambda *_args, **_kwargs: pytest.fail(
            'disabled frame reached image conversion'
        )
    )
    message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=1, nanosec=2),
            frame_id='camera',
        )
    )

    node_module.Sam3InferenceNode._on_image(node, message)

    assert released == [True]


def test_destroy_closes_runtime_once_before_ros_node(node_module, monkeypatch):
    """Close inference workers once before delegating ROS destruction."""
    events = []
    monkeypatch.setattr(
        node_module.Node,
        'destroy_node',
        lambda _node: events.append('destroy') or True,
    )
    node = object.__new__(node_module.Sam3InferenceNode)
    node._infer_lock = threading.Lock()
    node._runtime_closed = False
    node._live = SimpleNamespace(close=lambda: events.append('close'))
    node.get_logger = lambda: _Logger()

    assert node.destroy_node() is True
    assert node.destroy_node() is True

    assert events == ['close', 'destroy', 'destroy']


def test_closed_runtime_rejects_mutating_services(node_module):
    """Reject prompt and enable requests once runtime shutdown begins."""
    node = SimpleNamespace(
        _infer_lock=threading.Lock(),
        _runtime_closed=True,
        _enabled=True,
        _prompts=['floor'],
        _class_ids=[1],
        _prompt_to_class_id={'floor': 1},
        _live=SimpleNamespace(
            reset_prompts=lambda _prompts: pytest.fail(
                'closed runtime was mutated'
            )
        ),
        _validate_prompt_config=node_module.Sam3InferenceNode._validate_prompt_config,
        get_logger=lambda: _Logger(),
    )
    prompt_response = SimpleNamespace(success=None, message='')
    enable_response = SimpleNamespace(success=None, message='')

    node_module.Sam3InferenceNode._on_change_prompt(
        node,
        SimpleNamespace(prompts=['wall'], class_ids=[2]),
        prompt_response,
    )
    node_module.Sam3InferenceNode._on_enable(
        node,
        SimpleNamespace(data=False),
        enable_response,
    )

    assert prompt_response.success is False
    assert enable_response.success is False
    assert node._enabled is True
