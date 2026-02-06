FROM nvcr.io/nvidia/tritonserver:23.10-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /models

# ⬇️ здесь позже будет:
# aws s3 cp s3://YOUR_BUCKET/yolo/model.plan yolo/1/model.plan
# aws s3 cp s3://YOUR_BUCKET/yolo_pose/model.plan yolo_pose/1/model.plan

CMD ["tritonserver", \
     "--model-repository=/models", \
     "--grpc-port=8001", \
     "--http-port=8000", \
     "--metrics-port=8002"]

