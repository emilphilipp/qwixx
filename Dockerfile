FROM nginx:alpine
COPY index.html qwixx-engine.js /usr/share/nginx/html/
COPY table /usr/share/nginx/html/table
