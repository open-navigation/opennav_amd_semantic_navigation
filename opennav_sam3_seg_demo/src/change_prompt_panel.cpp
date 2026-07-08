#include <ament_index_cpp/get_package_share_directory.hpp>

#include "change_prompt_panel.hpp"

namespace opennav_sam3_seg_demo {

ChangePromptPanel::ChangePromptPanel()
: QObject {},
  widget_ {std::make_unique<QWidget>()},
  ui_ {std::make_unique<Ui::ChangePromptPanel>()}
{
  using PromptsValidator = QRegularExpressionValidator;
  
  widget_->setWindowFlags(
      Qt::Window 
    | Qt::CustomizeWindowHint
    | Qt::WindowTitleHint
    | Qt::WindowMinimizeButtonHint 
    | Qt::WindowStaysOnTopHint);
  
  ui_->setupUi(widget_.get());

  const std::string send_icon_path = 
    ament_index_cpp::get_package_share_directory("opennav_sam3_seg_demo") + "/icons/tick.svg";

  ui_->sendButton->setIcon(QPixmap(QString::fromStdString(send_icon_path)));

  const std::string main_icon_path =
    ament_index_cpp::get_package_share_directory("opennav_sam3_seg_demo") + "/icons/main_icon.png";
  
  widget_->setWindowIcon(QIcon(QString::fromStdString(main_icon_path)));

  esc_clear_shortcut_ = 
    std::make_unique<QShortcut>(QKeySequence(Qt::Key_Escape), ui_->promptsInput);
  
  esc_clear_shortcut_->setContext(Qt::WidgetWithChildrenShortcut);

  prompts_validator_ = 
    std::make_unique<PromptsValidator>(QRegularExpression("[A-Za-z, ]*"), ui_->promptsInput);

  ui_->promptsInput->setValidator(prompts_validator_.get());

  connect(esc_clear_shortcut_.get(), SIGNAL(activated()), ui_->promptsInput, SLOT(clear()));

  connect(ui_->sendButton, SIGNAL(clicked()), this, SLOT(applyPrompt()));
  connect(ui_->promptsInput, SIGNAL(returnPressed()), this, SLOT(applyPrompt()));

  connect(this, SIGNAL(panelDisabledChanged(bool)), ui_->promptsInput, SLOT(setDisabled(bool)));
  connect(this, SIGNAL(panelDisabledChanged(bool)), ui_->sendButton, SLOT(setDisabled(bool)));

  connect(this, SIGNAL(sendingDisabledChanged(bool)), ui_->sendButton, SLOT(setDisabled(bool)));

  connect(this, SIGNAL(statusLabelChanged(QString)), ui_->statusLabel, SLOT(setText(QString)));

  widget_->show();
  setupROS();
}

void ChangePromptPanel::setupROS()
{
  rclcpp_poll_timer_ = std::make_unique<QTimer>(this);
  connect(rclcpp_poll_timer_.get(), SIGNAL(timeout()), this, SLOT(rclcppPoll()));

  node_ = std::make_shared<rclcpp::Node>("change_prompt_node");
  
  timer_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  client_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  timer_ =
    node_->create_wall_timer(
      std::chrono::seconds(2),
      std::bind(&ChangePromptPanel::checkServiceAvailability, this),
      timer_cb_group_);

  change_prompt_client_ =
    node_->create_client<opennav_sam3_msgs::srv::ChangePrompt>(
      "/sam3_inference/change_prompt",
      rmw_qos_profile_services_default,
      client_cb_group_);

  service_request_ = std::make_shared<opennav_sam3_msgs::srv::ChangePrompt::Request>();

  executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
  executor_->add_node(node_);
  spin_thread_ = std::make_unique<std::thread>( [this](){ executor_->spin(); } );

  rclcpp_poll_timer_->start(500);

  service_available_ = change_prompt_client_->wait_for_service(std::chrono::seconds(1));

  if (service_available_)
    enablePanel();
  else
    disablePanel();

  RCLCPP_INFO(node_->get_logger(), "Panel initialized!");
}

ChangePromptPanel::~ChangePromptPanel()
{
  executor_->cancel();
  spin_thread_->join();
}

void ChangePromptPanel::applyPrompt()
{
  if (ui_->promptsInput->text().isEmpty()) {
    return;
  }

  if (!service_request_->prompts.empty()) {
    service_request_->prompts.clear();
  }

  if (!service_request_->class_ids.empty()) {
    service_request_->class_ids.clear();
  }

  std::uint16_t class_id {1};
  for (const auto & prompt : ui_->promptsInput->text().split(',', Qt::SkipEmptyParts)) {
    service_request_->prompts.push_back(prompt.toStdString());
    
    service_request_->class_ids.push_back(class_id);
    class_id += 1;
  }

  sendServiceRequest(service_request_);
}

void ChangePromptPanel::rclcppPoll()
{
  if (!rclcpp::ok()) {
    qApp->quit();
  }
}

void ChangePromptPanel::enablePanel()
{
  Q_EMIT statusLabelChanged("Type things to segment!");
  Q_EMIT panelDisabledChanged(false);
}

void ChangePromptPanel::disablePanel()
{
  Q_EMIT statusLabelChanged("Waiting for service...");
  Q_EMIT panelDisabledChanged(true);
}

void ChangePromptPanel::enableSending()
{
  connect(ui_->promptsInput, SIGNAL(returnPressed()), this, SLOT(applyPrompt()));
  Q_EMIT sendingDisabledChanged(false);
}

void ChangePromptPanel::disableSending()
{
  disconnect(ui_->promptsInput, SIGNAL(returnPressed()), this, SLOT(applyPrompt()));
  Q_EMIT sendingDisabledChanged(true);
}

void ChangePromptPanel::checkServiceAvailability()
{
  service_available_ = change_prompt_client_->wait_for_service(std::chrono::seconds(2));

  if (service_available_)
    enablePanel();
  else
    disablePanel();
}

void ChangePromptPanel::sendServiceRequest(
  const opennav_sam3_msgs::srv::ChangePrompt::Request::SharedPtr& request)
{
  using ServiceResponseFuture =
    rclcpp::Client<opennav_sam3_msgs::srv::ChangePrompt>::SharedFuture;

  disableSending();

  change_prompt_client_->async_send_request(
    request,
    [this](ServiceResponseFuture /*future*/) {
      enableSending();
    });
}

}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  QApplication app {argc, argv};

  auto panel = opennav_sam3_seg_demo::ChangePromptPanel();

  const int return_code = app.exec();

  rclcpp::shutdown();
  return return_code;
}
