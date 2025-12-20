#pragma once

#include <QMainWindow>
#include <QWidget>

namespace kelly {

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override = default;

private:
    void setupUi();
    void createMenus();
    void createToolbar();
};

} // namespace kelly
