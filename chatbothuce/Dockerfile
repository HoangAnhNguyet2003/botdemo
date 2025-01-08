# Step 1: Build the React app
FROM node:20-slim AS build

# Set working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json to install dependencies
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install

# Copy the entire project directory into the container
COPY . ./

# Build the React app for production
RUN npm run build

# Step 2: Serve the React app with Nginx
FROM nginx:stable-alpine

# Copy built React app from the previous stage to Nginx's web root
COPY --from=build /app/build /usr/share/nginx/html

# Copy custom Nginx configuration
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80 to the host
EXPOSE 80

# Start Nginx server
CMD ["nginx", "-g", "daemon off;"]
