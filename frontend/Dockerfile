# Use Node.js base
FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# Copy source code
COPY . .

# Expose Vite port
EXPOSE 5173

# Run dev server with host flag to allow access from outside container
CMD ["npm", "run", "dev", "--", "--host"]