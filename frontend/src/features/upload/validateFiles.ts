/** Returns an error message, or null when the selection is valid: one video, or one or more photos. */
export function validateFiles(files: File[]): string | null {
  if (files.length === 0) return 'Choose a video or a set of photos.'
  const videos = files.filter((f) => f.type.startsWith('video/')).length
  const images = files.filter((f) => f.type.startsWith('image/')).length
  if (videos + images !== files.length) return 'Only video and image files are supported.'
  if (videos > 0 && files.length > 1) return 'Upload either one video or several photos, not both.'
  return null
}
