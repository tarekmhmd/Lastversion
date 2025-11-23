#!/bin/bash

# SmartAiCity Setup Script
# This script automates the setup process for the Safe Smart City AI system

echo "🏙️  SmartAiCity Setup Script"
echo "================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running on Windows (WSL/Git Bash)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo -e "${YELLOW}Detected Windows environment${NC}"
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Check prerequisites
echo "Checking prerequisites..."

# Check Python
if command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python not found. Please install Python 3.10+"
    exit 1
fi

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_success "Node.js $NODE_VERSION found"
else
    print_error "Node.js not found. Please install Node.js 18+"
    exit 1
fi

# Check Redis
if command -v redis-server &> /dev/null; then
    print_success "Redis found"
else
    print_info "Redis not found. Installing via Docker is recommended."
fi

echo ""
echo "Choose installation method:"
echo "1) Docker (Recommended)"
echo "2) Manual Setup"
read -p "Enter your choice (1 or 2): " choice

if [ "$choice" == "1" ]; then
    echo ""
    print_info "Setting up with Docker..."

    # Check Docker
    if command -v docker &> /dev/null; then
        print_success "Docker found"
    else
        print_error "Docker not found. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        print_success "Docker Compose found"
    else
        print_error "Docker Compose not found. Please install Docker Compose first."
        exit 1
    fi

    echo ""
    print_info "Building Docker containers..."
    docker-compose build

    echo ""
    print_info "Starting services..."
    docker-compose up -d

    echo ""
    print_success "Setup complete!"
    echo ""
    echo "Services are running at:"
    echo "  Frontend: http://localhost:3000"
    echo "  Backend API: http://localhost:8000"
    echo "  Admin Panel: http://localhost:8000/admin"
    echo ""
    echo "To view logs: docker-compose logs -f"
    echo "To stop: docker-compose down"

elif [ "$choice" == "2" ]; then
    echo ""
    print_info "Setting up manually..."

    # Backend setup
    echo ""
    print_info "Setting up backend..."
    cd backend

    # Create virtual environment
    print_info "Creating Python virtual environment..."
    $PYTHON_CMD -m venv venv

    # Activate virtual environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi

    # Install Python dependencies
    print_info "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt

    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        print_info "Creating .env file..."
        cat > .env << EOL
SECRET_KEY=django-insecure-change-this-in-production
DEBUG=True
DATABASE_URL=sqlite:///db.sqlite3
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
EOL
    fi

    # Run migrations
    print_info "Running database migrations..."
    $PYTHON_CMD manage.py migrate

    # Create superuser (optional)
    echo ""
    read -p "Create admin superuser? (y/n): " create_user
    if [ "$create_user" == "y" ]; then
        $PYTHON_CMD manage.py createsuperuser
    fi

    # Collect static files
    print_info "Collecting static files..."
    $PYTHON_CMD manage.py collectstatic --noinput

    cd ..

    # Frontend setup
    echo ""
    print_info "Setting up frontend..."
    cd frontend

    # Install Node dependencies
    print_info "Installing Node.js dependencies..."
    npm install

    # Create .env.local file
    if [ ! -f .env.local ]; then
        print_info "Creating .env.local file..."
        echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
    fi

    cd ..

    echo ""
    print_success "Setup complete!"
    echo ""
    echo "To run the application:"
    echo ""
    echo "Terminal 1 - Backend:"
    echo "  cd backend"
    echo "  source venv/bin/activate  (or venv\\Scripts\\activate on Windows)"
    echo "  python manage.py runserver"
    echo ""
    echo "Terminal 2 - Celery Worker:"
    echo "  cd backend"
    echo "  source venv/bin/activate"
    echo "  celery -A config worker -l info"
    echo ""
    echo "Terminal 3 - Frontend:"
    echo "  cd frontend"
    echo "  npm run dev"
    echo ""
    echo "Access the application at http://localhost:3000"

else
    print_error "Invalid choice. Exiting."
    exit 1
fi

echo ""
echo "================================"
echo "🎉 SmartAiCity is ready!"
echo "================================"
