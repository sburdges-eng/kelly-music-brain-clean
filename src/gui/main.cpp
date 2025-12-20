#include <QApplication>
#include "main_window.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    
    kelly::MainWindow window;
    window.setWindowTitle("Kelly - Therapeutic iDAW");
    window.resize(1024, 768);
    window.show();
    
    return app.exec();
}
