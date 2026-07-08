#pragma once

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include <QtWidgets>
#include <ui_change_prompt_panel.h>

#include <opennav_sam3_msgs/srv/change_prompt.hpp>

namespace opennav_sam3_seg_demo {

class ChangePromptPanel : public QObject {

Q_OBJECT
public:
  ChangePromptPanel();
  ~ChangePromptPanel();

  void setupROS();

public Q_SLOTS:
  void applyPrompt();
  void rclcppPoll();

Q_SIGNALS:
  void panelDisabledChanged(bool state);
  void sendingDisabledChanged(bool state);
  void statusLabelChanged(QString newLabel);

private:
  void enablePanel();
  void disablePanel();
  void enableSending();
  void disableSending();
  void checkServiceAvailability();

  void sendServiceRequest(const opennav_sam3_msgs::srv::ChangePrompt::Request::SharedPtr& request);

  bool service_available_ {false};

  std::unique_ptr<QWidget> widget_ {nullptr};
  std::unique_ptr<Ui::ChangePromptPanel> ui_ {nullptr};

  rclcpp::Node::SharedPtr node_ {nullptr};

  rclcpp::executors::MultiThreadedExecutor::SharedPtr executor_ {nullptr};
  std::unique_ptr<std::thread> spin_thread_ {nullptr};

  rclcpp::CallbackGroup::SharedPtr timer_cb_group_ {nullptr};
  rclcpp::CallbackGroup::SharedPtr client_cb_group_ {nullptr};

  rclcpp::TimerBase::SharedPtr timer_ {nullptr};

  rclcpp::Client<opennav_sam3_msgs::srv::ChangePrompt>::SharedPtr change_prompt_client_ {nullptr};
  opennav_sam3_msgs::srv::ChangePrompt::Request::SharedPtr service_request_ {nullptr};

  std::unique_ptr<QShortcut> esc_clear_shortcut_ {nullptr};
  std::unique_ptr<QRegularExpressionValidator> prompts_validator_ {nullptr};
  std::unique_ptr<QTimer> rclcpp_poll_timer_ {nullptr};
};

}
