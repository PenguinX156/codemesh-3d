$env:VITE_API_BASE_URL = "http://localhost:8010/api"
Set-Location "$PSScriptRoot\..\frontend"
npm run dev -- --port 5180 --strictPort
