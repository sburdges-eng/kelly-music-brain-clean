# Deployment Guide

This guide covers deploying the Music Brain API to production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Configuration](#environment-configuration)
3. [Docker Deployment](#docker-deployment)
4. [Production Considerations](#production-considerations)
5. [Monitoring and Logging](#monitoring-and-logging)
6. [Scaling](#scaling)
7. [Security Checklist](#security-checklist)

## Prerequisites

- Docker and Docker Compose installed
- Python 3.11+ (for local development)
- A secure secret key for JWT tokens
- Domain name and SSL certificate (for production)

## Environment Configuration

### 1. Create Environment File

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

### 2. Configure Security Settings

**CRITICAL**: Update these values before deploying:

```bash
# Generate a secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Add to .env
SECRET_KEY=your-generated-secret-key-here
```

### 3. Configure CORS

Update CORS origins to match your frontend domain:

```bash
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### 4. Configure Rate Limits

Adjust rate limits based on your expected traffic:

```bash
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
RATE_LIMIT_PER_DAY=10000
```

## Docker Deployment

### Quick Start

```bash
# Build and start the API
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop the API
docker-compose down
```

### Production Build

```bash
# Build the image
docker build -t music-brain-api:latest .

# Run the container
docker run -d \
  --name music_brain_api \
  -p 8000:8000 \
  --env-file .env \
  music-brain-api:latest
```

### Docker Compose Production

For production, use a production-optimized `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    restart: always
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./emotion_thesaurus:/app/emotion_thesaurus:ro
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

## Production Considerations

### 1. Reverse Proxy

Use Nginx or Traefik as a reverse proxy:

**Nginx Configuration** (`/etc/nginx/sites-available/music-brain`):

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. SSL/TLS

Use Let's Encrypt for free SSL certificates:

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d api.yourdomain.com
```

### 3. Process Management

For non-Docker deployments, use systemd:

**Service File** (`/etc/systemd/system/music-brain-api.service`):

```ini
[Unit]
Description=Music Brain API
After=network.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/app
Environment="PATH=/app/.venv/bin"
ExecStart=/app/.venv/bin/uvicorn music_brain.api:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable music-brain-api
sudo systemctl start music-brain-api
sudo systemctl status music-brain-api
```

## Monitoring and Logging

### Health Checks

The API includes a health check endpoint:

```bash
curl http://localhost:8000/health
```

### Metrics Endpoint

Access metrics (requires authentication):

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/metrics
```

### Logging

Logs are output to stdout/stderr and can be collected by:

- **Docker**: `docker-compose logs -f`
- **Systemd**: `journalctl -u music-brain-api -f`
- **Log aggregation**: Configure log forwarding to ELK, Loki, or similar

### External Monitoring

Consider integrating:

- **Sentry**: Error tracking
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **Datadog/New Relic**: APM and monitoring

## Scaling

### Horizontal Scaling

1. **Load Balancer**: Use Nginx or HAProxy to distribute traffic
2. **Multiple Instances**: Run multiple API containers behind the load balancer
3. **Shared State**: Use Redis for distributed rate limiting

### Vertical Scaling

Adjust resource limits in Docker Compose:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 4G
```

### Database Considerations

If implementing user management, use a production database:

- **PostgreSQL**: Recommended for production
- **Connection Pooling**: Use PgBouncer or similar
- **Backups**: Implement regular backup strategy

## Security Checklist

- [ ] Change `SECRET_KEY` from default value
- [ ] Configure `CORS_ORIGINS` to specific domains (not `*`)
- [ ] Enable rate limiting
- [ ] Use HTTPS in production
- [ ] Set up firewall rules
- [ ] Regularly update dependencies
- [ ] Implement authentication for sensitive endpoints
- [ ] Set up log monitoring for suspicious activity
- [ ] Configure security headers (already included in middleware)
- [ ] Use non-root user in Docker containers (already configured)
- [ ] Regularly rotate API keys and tokens
- [ ] Implement request validation
- [ ] Set up DDoS protection (Cloudflare, AWS Shield, etc.)

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs api

# Check container status
docker ps -a

# Restart container
docker-compose restart api
```

### Port Already in Use

```bash
# Find process using port
lsof -i :8000

# Kill process or change PORT in .env
```

### Permission Errors

```bash
# Fix file permissions
sudo chown -R appuser:appuser /app
```

### High Memory Usage

- Check for memory leaks
- Reduce `RATE_LIMIT_PER_DAY` if needed
- Increase container memory limits

## Next Steps

- Set up CI/CD pipeline
- Configure automated backups
- Implement monitoring alerts
- Set up staging environment
- Create runbooks for common issues
