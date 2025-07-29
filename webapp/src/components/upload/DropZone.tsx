import React, { useRef } from 'react';
import { Button } from '../ui/Button';

interface DropZoneProps {
  accept?: string[];
  maxSize?: number; // MB
  onUpload: (file: File) => void;
  preview?: boolean;
}

export const DropZone: React.FC<DropZoneProps> = ({ accept = ['csv', 'xlsx', 'json'], maxSize = 100, onUpload, preview }) => {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFile = (file: File) => {
    const ext = file.name.split('.').pop()?.toLowerCase();
    if (!accept.includes(ext || '')) {
      alert('Format non supporté.');
      return;
    }
    if (file.size > maxSize * 1024 * 1024) {
      alert('Fichier trop volumineux.');
      return;
    }
    onUpload(file);
  };

  return (
    <div
      className="border-2 border-dashed border-blue-300 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer bg-blue-50 hover:bg-blue-100 transition"
      onClick={() => inputRef.current?.click()}
      onDrop={handleDrop}
      onDragOver={e => e.preventDefault()}
    >
      <span className="text-blue-500 text-2xl mb-2">📁</span>
      <span className="mb-2 text-gray-700">Glissez-déposez ou cliquez pour sélectionner un fichier</span>
      <input
        ref={inputRef}
        type="file"
        accept={accept.map(a => '.' + a).join(',')}
        className="hidden"
        onChange={e => e.target.files && handleFile(e.target.files[0])}
      />
      <Button variant="ghost" size="sm" className="mt-2">Choisir un fichier</Button>
    </div>
  );
};
