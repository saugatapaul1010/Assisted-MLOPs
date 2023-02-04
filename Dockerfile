# Dockerfile

# FROM directive instructing base image to build upon
#FROM python:3.8.5

FROM az-cli-az-ml

ENV ENV=TEST
# copy source and install dependencies
RUN mkdir -p /opt/app
RUN mkdir -p /opt/app/pip_cache
RUN mkdir -p /opt/app/azure_assisted_mlops_backend
COPY requirements.txt /opt/app/

# COPY pip_cache /opt/app/pip_cache/
COPY . /opt/app/azure_assisted_mlops_backend/
RUN mkdir -p /opt/app/azure_assisted_mlops_backend/uploads
RUN mkdir -p /opt/app/azure_assisted_mlops_backend/static

RUN mkdir -p /opt/app/azure_assisted_mlops_backend/yaml_outputs /opt/app/azure_assisted_mlops_backend/yaml_outputs/training
RUN mkdir -p /opt/app/azure_assisted_mlops_backend/yaml_templates /opt/app/azure_assisted_mlops_backend/yaml_templates/deployment /opt/app/azure_assisted_mlops_backend/yaml_templates/training
RUN mkdir -p /root/.azure-devops
RUN mkdir -p /root/.config
RUN mkdir -p /root/.azure/
RUN touch /root/.azure/msal_token_cache.json

WORKDIR /opt/app
RUN pip install -r requirements.txt
RUN chown -R www-data:www-data /opt/app
WORKDIR /opt/app/azure_assisted_mlops_backend
#RUN python manage.py makemigrations
#RUN python manage.py migrate



#For giving permission to root inside container
RUN chown -R www-data:www-data /root

#For giving permission to tmp inside container
RUN chown -R www-data:www-data /tmp

#For giving permission to usr inside container
RUN chown -R www-data:www-data /usr

#For giving permission to home inside container
RUN chown -R www-data:www-data /home

#For giving permission to opt inside container
RUN chown -R www-data:www-data /opt

#For giving permission to /root/.azure-devops inside container
RUN chown -R www-data:www-data /root/.azure-devops

#For giving permission to /root/.config inside container
RUN chown -R www-data:www-data /root/.config

#For endpoints
RUN chown -R www-data:www-data /root/.azure/
RUN chown www-data:www-data /root/.azure/msal_token_cache.json

# start server #Map the port to 8001 in host VM to avoid conflict
EXPOSE 8001
STOPSIGNAL SIGTERM
WORKDIR /opt/app/azure_assisted_mlops_backend

RUN chmod +x startup.sh
RUN ./startup.sh
CMD ["gunicorn","mlops_apis.wsgi:application","--user","www-data","--bind","0.0.0.0:8001","--workers","4","--timeout","120"]
# CMD ["celery","-A","assisted_mlops_backend","worker","-l","info","-P","solo","-E","-Q","async_q","--concurrency=500","--detach"]
#os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'