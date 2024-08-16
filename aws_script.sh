git pull https://github.com/zollyzeus/ASR-Deployment.git
docker rm $(docker ps -a -f status=exited -q)
docker rmi $(docker images -a -q)
docker builder prune -a
docker build -t asrproject .
docker run -p 5000:5000 asrproject