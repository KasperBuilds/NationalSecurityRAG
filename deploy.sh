#!/bin/bash
# ==============================================================
# NationalSecurityRAG - DigitalOcean Droplet Deployment Script
# Run this AFTER SSH-ing into your Droplet
# Usage: bash deploy.sh
# ==============================================================

set -e

# Detect the non-root user (DigitalOcean uses 'root' by default,
# but you may have created a user like 'deploy' or 'ubuntu')
APP_USER=${DEPLOY_USER:-root}
APP_HOME=$(eval echo ~$APP_USER)

echo "========================================="
echo "  NationalSecurityRAG Droplet Setup"
echo "  User: $APP_USER | Home: $APP_HOME"
echo "========================================="

# --- 1. System dependencies ---
echo "📦 Installing system dependencies..."
sudo apt update -y
sudo apt install -y python3-pip python3-venv git nginx certbot python3-certbot-nginx

# --- 2. Clone repo ---
echo "📥 Cloning repository..."
cd $APP_HOME
if [ -d "NationalSecurityRAG" ]; then
    echo "   Repo already exists, pulling latest..."
    cd NationalSecurityRAG && git pull
else
    git clone https://github.com/KasperBuilds/NationalSecurityRAG.git
    cd NationalSecurityRAG
fi

# --- 3. Python environment ---
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# --- 4. Environment variables ---
echo "🔑 Setting up environment variables..."
if [ ! -f /etc/nationalsecurityrag.env ]; then
    echo "   Creating env file - YOU MUST EDIT THIS with your API key!"
    sudo tee /etc/nationalsecurityrag.env > /dev/null <<'ENVFILE'
OPENROUTER_API_KEY=sk-or-your-openrouter-key-here
CHROMA_PATH=./chroma_db
ENVFILE
    sudo chmod 600 /etc/nationalsecurityrag.env
    echo "   ⚠️  Edit /etc/nationalsecurityrag.env with your real API keys!"
    echo "   Run: sudo nano /etc/nationalsecurityrag.env"
else
    echo "   Env file already exists, skipping..."
fi

# --- 5. Systemd service ---
echo "⚙️  Creating systemd service..."
sudo tee /etc/systemd/system/nssrag.service > /dev/null <<'SERVICE'
[Unit]
Description=National Security RAG FastAPI App
After=network.target

[Service]
User=$APP_USER
WorkingDirectory=$APP_HOME/NationalSecurityRAG
EnvironmentFile=/etc/nationalsecurityrag.env
ExecStart=$APP_HOME/NationalSecurityRAG/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable nssrag

# --- 6. Nginx reverse proxy ---
echo "🌐 Configuring Nginx..."
sudo tee /etc/nginx/sites-available/nssrag > /dev/null <<'NGINX'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_connect_timeout 10s;
    }
}
NGINX

sudo ln -sf /etc/nginx/sites-available/nssrag /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

# --- 7. Start the app ---
echo "🚀 Starting the application..."
sudo systemctl start nssrag

echo ""
echo "========================================="
echo "  ✅ Deployment Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Edit API keys:  sudo nano /etc/nationalsecurityrag.env"
echo "  2. Restart app:    sudo systemctl restart nssrag"
echo "  3. Check status:   sudo systemctl status nssrag"
echo "  4. View logs:      sudo journalctl -u nssrag -f"
echo "  5. Your site:      http://$(curl -s ifconfig.me 2>/dev/null || echo '<your-droplet-ip>')"
echo ""
echo "To add HTTPS with a custom domain:"
echo "  1. Point your domain's A record to this Droplet's IP"
echo "  2. Edit /etc/nginx/sites-available/nssrag → change server_name to your domain"
echo "  3. Run: sudo certbot --nginx -d yourdomain.com"
echo ""
