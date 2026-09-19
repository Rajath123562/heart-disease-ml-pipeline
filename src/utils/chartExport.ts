/**
 * Utility to export SVG / Chart container to downloadable PNG image
 */

export async function downloadChartAsPng(containerId: string, filename: string = 'chart.png') {
  try {
    const container = document.getElementById(containerId);
    if (!container) return;

    const svgElement = container.querySelector('svg');
    if (!svgElement) return;

    const svgData = new XMLSerializer().serializeToString(svgElement);
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const svgSize = svgElement.getBoundingClientRect();
    const scale = 2; // High DPI
    canvas.width = (svgSize.width || 600) * scale;
    canvas.height = (svgSize.height || 400) * scale;

    const img = new Image();
    const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
    const URL = window.URL || window.webkitURL || window;
    const blobURL = URL.createObjectURL(svgBlob);

    img.onload = () => {
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(blobURL);

      const pngUrl = canvas.toDataURL('image/png');
      const downloadLink = document.createElement('a');
      downloadLink.href = pngUrl;
      downloadLink.download = filename;
      document.body.appendChild(downloadLink);
      downloadLink.click();
      document.body.removeChild(downloadLink);
    };

    img.src = blobURL;
  } catch (err) {
    console.error('Failed to export chart PNG:', err);
  }
}
