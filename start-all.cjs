// Script pour démarrer backend (FastAPI) + frontend (Vite) depuis la racine
const { spawn } = require('child_process');
const path = require('path');

// Détection multiplateforme pour le chemin Python
const backendCmd = process.platform === 'win32'
  ? '.venv\\Scripts\\python.exe'
  : './.venv/bin/python';
const backendArgs = ['-m', 'uvicorn', 'api.main:app', '--reload', '--port', '8000'];

const frontendCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const frontendArgs = ['run', 'dev'];
const frontendCwd = path.join(__dirname, 'webapp');

const backend = spawn(backendCmd, backendArgs, { stdio: 'inherit' });
const frontend = spawn(frontendCmd, frontendArgs, { cwd: frontendCwd, stdio: 'inherit' });

function shutdown() {
  backend.kill();
  frontend.kill();
  process.exit();
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

backend.on('close', code => {
  console.log(`FastAPI backend exited with code ${code}`);
});
frontend.on('close', code => {
  console.log(`Frontend exited with code ${code}`);
});
