# Mental Health Chatbot

This project is a mental health chatbot that utilizes the Ollama API for providing mental health support and resources. It is structured into a server and client application, allowing for a seamless interaction between users and the chatbot.

## Project Structure

```
mental-health-chatbot
├── server                # Backend server application
│   ├── src
│   │   ├── index.ts     # Entry point of the server application
│   │   ├── routes       # Contains route definitions
│   │   │   └── chat.ts  # Chat-related routes
│   │   ├── controllers  # Contains request handling logic
│   │   │   └── chatController.ts # Chat controller
│   │   ├── services     # Contains service logic
│   │   │   └── ollama.ts # Ollama API interaction
│   │   ├── middleware   # Middleware functions
│   │   │   └── rateLimit.ts # Rate limiting middleware
│   │   └── types        # Type definitions
│   │       └── index.d.ts # TypeScript interfaces
│   ├── package.json     # NPM configuration for the server
│   ├── tsconfig.json    # TypeScript configuration for the server
│   └── Dockerfile       # Docker configuration for the server
├── client                # Frontend client application
│   ├── src
│   │   ├── App.tsx      # Main React component
│   │   ├── components    # UI components
│   │   │   └── ChatUI.tsx # Chat user interface
│   │   └── services     # API service logic
│   │       └── api.ts   # API calls to the server
│   ├── package.json     # NPM configuration for the client
│   └── tsconfig.json    # TypeScript configuration for the client
├── notebooks             # Jupyter notebooks for analysis
│   └── Mental-health.ipynb # Notebook related to mental health
├── render.yaml           # Deployment configuration for Render
├── .gitignore            # Git ignore file
└── README.md             # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd mental-health-chatbot
   ```

2. **Install server dependencies:**
   Navigate to the `server` directory and run:
   ```bash
   npm install
   ```

3. **Install client dependencies:**
   Navigate to the `client` directory and run:
   ```bash
   npm install
   ```

4. **Run the server:**
   From the `server` directory, start the server:
   ```bash
   npm start
   ```

5. **Run the client:**
   From the `client` directory, start the client:
   ```bash
   npm start
   ```

## Usage

Once both the server and client are running, you can access the chatbot interface through your web browser. The chatbot will provide mental health support and resources based on user interactions.

## Deployment

This project can be deployed on Render using the provided `render.yaml` configuration file. Follow the Render documentation for deploying Node.js applications.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.