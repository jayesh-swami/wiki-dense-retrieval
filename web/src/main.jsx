import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./index.css";

// No StrictMode: in dev it runs effects twice, which would send the ?q= search (and its
// Cohere call) twice.
createRoot(document.getElementById("root")).render(<App />);
