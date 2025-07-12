with import <nixpkgs> {};
stdenv.mkDerivation {
  name = "myenv";
  buildInputs = [
    python3
    python3Packages.pymupdf
    python3Packages.langchain
    python3Packages.langchain-community
    python3Packages.langchain-core
    python3Packages.langchain-ollama
    python3Packages.chromadb
    python3Packages.pytesseract
    python3Packages.pillow
    python3Packages.fastapi
    python3Packages.uvicorn
    python3Packages.python-multipart
  ];
}
