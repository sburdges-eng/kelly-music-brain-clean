# Docker Installation Guide for macOS

## ✅ Installation Status

Docker Desktop is **already installed** on your system at `/Applications/Docker.app`.

## 🚀 Quick Start

### 1. Start Docker Desktop

Docker Desktop should be starting automatically. If not:

```bash
# Open Docker Desktop
open -a Docker

# Or manually:
# 1. Open Applications folder
# 2. Double-click Docker.app
# 3. Wait for the whale icon to appear in the menu bar
```

### 2. Verify Installation

Run the verification script:

```bash
cd ml_training
./verify-docker.sh
```

This will check:
- ✓ Docker Desktop installation
- ✓ Docker CLI availability
- ✓ Docker Compose availability
- ✓ Docker daemon status
- ⚠ GPU support (not available on macOS)

### 3. Wait for Docker to Start

Docker Desktop can take 30-60 seconds to fully start. You'll know it's ready when:
- The whale icon appears in the menu bar (top right)
- The icon stops animating
- `docker --version` works in terminal

### 4. Test the Setup

```bash
cd ml_training

# Check status
./docker-run.sh status

# Build images (first time)
./docker-run.sh build

# Start training
./docker-run.sh start
```

---

## 📋 Manual Verification

If the script doesn't work, verify manually:

```bash
# Check Docker version
docker --version

# Check Docker Compose
docker compose version
# or
docker-compose --version

# Check if Docker is running
docker info

# Test with a simple container
docker run hello-world
```

---

## ⚠️ Important Notes for macOS

### GPU Support
- **macOS does NOT support GPU passthrough** in Docker
- Training will use **CPU** (slower but works)
- For GPU training, use:
  - Google Colab (free GPU)
  - AWS SageMaker
  - Linux machine with NVIDIA GPU

### Performance
- CPU training is **10-20x slower** than GPU
- Full training (5 models) may take **48-72 hours** on CPU
- Consider using cloud services for actual training

### Resource Usage
- Docker Desktop uses significant RAM (2-4GB)
- Ensure you have at least 8GB free RAM
- Close other applications if needed

---

## 🔧 Troubleshooting

### Docker CLI Not Found

**Problem**: `docker: command not found`

**Solution**:
1. Ensure Docker Desktop is running
2. Restart your terminal
3. Check PATH: `echo $PATH`
4. Docker Desktop should add `/usr/local/bin` to PATH automatically

### Docker Daemon Not Running

**Problem**: `Cannot connect to the Docker daemon`

**Solution**:
1. Open Docker Desktop manually
2. Wait for it to fully start (whale icon in menu bar)
3. Check Docker Desktop status in menu bar
4. Try: `docker info` to verify connection

### Permission Denied

**Problem**: `permission denied while trying to connect to the Docker daemon socket`

**Solution**:
```bash
# Add your user to docker group (if on Linux)
sudo usermod -aG docker $USER

# On macOS, Docker Desktop handles permissions automatically
# Just ensure Docker Desktop is running
```

### Out of Memory

**Problem**: Container runs out of memory

**Solution**:
1. Increase Docker Desktop memory:
   - Docker Desktop → Settings → Resources
   - Increase Memory to 4GB or more
2. Reduce batch size in training:
   ```bash
   ./docker-run.sh shell
   # Inside container:
   python train_all_models.py --batch-size 32  # Instead of 64
   ```

### Port Already in Use

**Problem**: `port is already allocated` (e.g., port 6006 for TensorBoard)

**Solution**:
```bash
# Find what's using the port
lsof -i :6006

# Kill the process or change port in docker-compose.yml
```

---

## 🎯 Next Steps

Once Docker is verified:

1. **Build the images**:
   ```bash
   ./docker-run.sh build
   ```

2. **Prepare datasets** (if needed):
   ```bash
   ./docker-run.sh data
   ```

3. **Start training**:
   ```bash
   # Foreground (see logs)
   ./docker-run.sh start

   # Background
   ./docker-run.sh start --detach
   ```

4. **Monitor progress**:
   ```bash
   # View logs
   ./docker-run.sh logs -f

   # Start TensorBoard
   ./docker-run.sh monitor
   # Open: http://localhost:6006
   ```

---

## 📚 Additional Resources

- [Docker Desktop for Mac Documentation](https://docs.docker.com/desktop/mac/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [DOCKER_QUICKSTART.md](./DOCKER_QUICKSTART.md) - Quick reference
- [README_DOCKER.md](./README_DOCKER.md) - Full documentation

---

## 💡 Tips

1. **Keep Docker Desktop running** - Don't quit it while using containers
2. **Check menu bar** - Whale icon shows Docker Desktop status
3. **Use helper script** - `./docker-run.sh` makes everything easier
4. **Monitor resources** - Docker Desktop shows CPU/RAM usage
5. **Clean up regularly** - Run `./docker-run.sh clean` to free space

---

## 🆘 Still Having Issues?

1. **Restart Docker Desktop**:
   - Quit Docker Desktop (menu bar → Quit)
   - Wait 10 seconds
   - Open Docker Desktop again

2. **Restart Terminal**:
   - Close and reopen terminal
   - Check PATH: `echo $PATH`

3. **Check Docker Desktop Logs**:
   - Docker Desktop → Troubleshoot → View logs

4. **Reinstall Docker Desktop** (last resort):
   ```bash
   brew uninstall --cask docker
   brew install --cask docker
   ```
