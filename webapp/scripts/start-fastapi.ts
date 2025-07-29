import { spawn } from 'child_process';
import net from 'net';

const PORT = 8000;
const HOST = '127.0.0.1';

function isApiRunning(host: string, port: number): Promise<boolean> {
  return new Promise(resolve => {
    const socket = new net.Socket();
    socket.setTimeout(500);
    socket.once('error', () => resolve(false));
    socket.once('timeout', () => resolve(false));
    socket.connect(port, host, () => {
      socket.end();
      resolve(true);
    });
  });
}

async function startFastAPI() {
  const running = await isApiRunning(HOST, PORT);
  if (running) {
    console.log('✅ FastAPI déjà démarré sur le port', PORT);
    return;
  }
  console.log('🚀 Démarrage de FastAPI...');
  const proc = spawn('uvicorn', ['api.main:app', '--host', HOST, '--port', PORT.toString(), '--reload'], {
    stdio: 'inherit',
    cwd: process.cwd(),
    shell: true,
  });
  proc.on('error', err => {
    console.error('Erreur au lancement de FastAPI:', err);
  });
}

startFastAPI();
