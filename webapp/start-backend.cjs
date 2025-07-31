// Script Node.js pour démarrer FastAPI automatiquement (CommonJS)
const { spawn } = require('child_process');

const backend = spawn('../venv/bin/python', ['-m', 'uvicorn', 'api.main:app', '--reload', '--port', '8000'], {
  cwd: '../',
  stdio: 'inherit',
  shell: true,
});

backend.on('close', (code) => {
  console.log(`FastAPI backend exited with code ${code}`);
});
