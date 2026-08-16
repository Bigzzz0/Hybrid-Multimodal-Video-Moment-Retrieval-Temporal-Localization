import os
from fastapi import Request, HTTPException, status
from fastapi.responses import StreamingResponse

def range_requests_response(
    request: Request, file_path: str, content_type: str = "video/mp4"
) -> StreamingResponse:
    """Handles HTTP 206 Byte-Range Requests for smooth seeking in HTML5 Video Player."""
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Video file not found."
        )

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("range")

    if range_header:
        # Range header format: "bytes=start-end"
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0])
        end = int(byte_range[1]) if byte_range[1] else file_size - 1
        
        # Clamp bounds
        start = max(0, start)
        end = min(file_size - 1, end)
        content_length = (end - start) + 1

        def iterfile():
            with open(file_path, "rb") as f:
                f.seek(start)
                bytes_remaining = content_length
                chunk_size = 1024 * 512 # 512 KB chunks
                while bytes_remaining > 0:
                    read_size = min(chunk_size, bytes_remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    bytes_remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": content_type,
        }
        return StreamingResponse(
            iterfile(), status_code=status.HTTP_206_PARTIAL_CONTENT, headers=headers
        )
    else:
        # Full content
        def iterfile_full():
            with open(file_path, "rb") as f:
                while chunk := f.read(1024 * 512):
                    yield chunk

        headers = {
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Content-Type": content_type,
        }
        return StreamingResponse(iterfile_full(), status_code=status.HTTP_200_OK, headers=headers)
