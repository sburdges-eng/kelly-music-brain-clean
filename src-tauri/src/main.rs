// Prevents additional console window on Windows
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod bridge;

use commands::{generate_music, interrogate, get_emotions, health_check, is_api_available};

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            generate_music, interrogate, get_emotions, health_check, is_api_available,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
