// Build: `npm run build` -> ui/dist, which the master serves at http://localhost:8000/.
// Develop: `npm run dev` -> http://localhost:5173, passing API calls on to the master on port 8000.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const master = process.env.MASTER_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/v1": master, "/api": master, "/health": master } },
});
