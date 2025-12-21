docker build -f docker/Dockerfile -t classix-jupyter .
docker run --rm -p 8888:8888 classix-jupyter
