// Image attachments: read a File into a data URL, downscaling large images
// so a screenshot does not become a multi-megabyte request body.

const MAX_SIDE = 1536;
const RESIZE_ABOVE_BYTES = 400 * 1024;
const JPEG_QUALITY = 0.9;

export async function readImageFile(file: File): Promise<string> {
  const original = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("Could not read image"));
    reader.readAsDataURL(file);
  });
  if (file.size <= RESIZE_ABOVE_BYTES || file.type === "image/gif") return original;
  try {
    const image = await loadImage(original);
    const scale = Math.min(1, MAX_SIDE / Math.max(image.width, image.height));
    if (scale >= 1 && file.size <= RESIZE_ABOVE_BYTES * 2) return original;
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(image.width * scale));
    canvas.height = Math.max(1, Math.round(image.height * scale));
    const context = canvas.getContext("2d");
    if (!context) return original;
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", JPEG_QUALITY);
  } catch {
    return original;
  }
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Could not decode image"));
    image.src = url;
  });
}

export function imageFilesFrom(items: DataTransferItemList | FileList | null | undefined): File[] {
  if (!items) return [];
  const files: File[] = [];
  for (const item of Array.from(items as ArrayLike<DataTransferItem | File>)) {
    if (item instanceof File) {
      if (item.type.startsWith("image/")) files.push(item);
    } else if (item.kind === "file" && item.type.startsWith("image/")) {
      const file = item.getAsFile();
      if (file) files.push(file);
    }
  }
  return files;
}
