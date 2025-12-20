import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import ErrorBoundary from "./ErrorBoundary";
import "./index.css";

console.log("main.tsx: Starting React app initialization");

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("Root element not found. Make sure index.html has a div with id='root'");
}

console.log("main.tsx: Root element found, creating React root");

try {
  const root = ReactDOM.createRoot(rootElement);
  console.log("main.tsx: React root created, rendering App");
  
  root.render(
    <React.StrictMode>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </React.StrictMode>,
  );
  
  console.log("main.tsx: App rendered successfully");
} catch (error) {
  console.error("main.tsx: Fatal error during render:", error);
  rootElement.innerHTML = `
    <div style="padding: 20px; font-family: monospace;">
      <h1>Fatal Error</h1>
      <p>${error instanceof Error ? error.message : String(error)}</p>
      <pre style="background: #f0f0f0; padding: 10px; overflow: auto;">
        ${error instanceof Error ? error.stack : String(error)}
      </pre>
    </div>
  `;
}
