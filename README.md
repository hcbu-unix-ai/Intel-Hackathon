# Intel-Hackathon

## How to run this application
```bash
cd app/
pip install -r requirements.txt
./entrypoint.sh
```

## Deploy using Docker
```bash
docker run -it -p 8505:8505 -v ./data/:/openvino/app/data/ hcbuunixai/openvino-chat:v1
```

## Deploy in Openshift
```bash
oc apply -f deploy.yaml

# To get application URL.
oc get route openvino-chat-route -o jsonpath='{.spec.host}'
```