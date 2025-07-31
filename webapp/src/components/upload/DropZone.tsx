import { useRef, useState } from 'react'
import { Button } from '../ui/Button'

interface DropZoneProps {
  accept?: string[]
  maxSize?: number // MB
  onUpload: (file: File) => void
  preview?: boolean
}

export function DropZone({ accept = ['csv', 'xlsx', 'json'], maxSize = 100, onUpload, preview }: DropZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }

  const handleFile = (file?: File) => {
    if (!file) return
    const ext = file.name.split('.').pop()?.toLowerCase() || ''
    if (!accept.includes(ext)) {
      setError(`Format non supporté: .${ext}`)
      return
    }
    if (file.size / 1024 / 1024 > maxSize) {
      setError(`Fichier trop volumineux (> ${maxSize}MB)`)
      return
    }
    setError(null)
    setSelectedFile(file)
    onUpload(file)
  }

  return (
    <div
      className="border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center cursor-pointer hover:bg-gray-50 w-full max-w-xl mx-auto"
      onClick={() => inputRef.current?.click()}
      onDrop={handleDrop}
      onDragOver={e => e.preventDefault()}
      tabIndex={0}
      aria-label="Zone de dépôt de fichier"
    >
      <span className="mb-2 text-gray-500">Glissez-déposez un fichier ou cliquez</span>
      <Button>Sélectionner un fichier</Button>
      <input
        ref={inputRef}
        type="file"
        accept={accept.map(a => '.' + a).join(',')}
        className="hidden"
        onChange={e => handleFile(e.target.files?.[0])}
      />
      {error && <div className="mt-2 text-red-500 text-sm">{error}</div>}
      {preview && selectedFile && (
        <div className="mt-4 text-xs text-gray-600">
          <strong>Fichier sélectionné :</strong> {selectedFile.name} ({(selectedFile.size/1024/1024).toFixed(2)} MB)
        </div>
      )}
    </div>
  )
}
