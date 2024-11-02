# Intel-Hackathon

## How to run this application
```bash
cd app/
pip install -r requirements.txt
./entrypoint.sh
```

## Deploy using Docker
```bash
mkdir data; chmod 0777 data
docker run  --rm -itd -p 8505:8505 --name openvino -v ./data/:/openvino/app/data/ hcbuunixai/openvino-chat:v0.1.1
```

## Deploy in Openshift
```bash
oc apply -f deploy.yaml

# To get application URL.
oc get route openvino-chat-route -o jsonpath='{.spec.host}'
```
