const connectWebSocket = (url: string, onMessage: (data: any) => void) => {
  const socket = new WebSocket(url);

  socket.onopen = () => {
      console.log(`🔌 Connected to WebSocket: ${url}`);
  };

  socket.onmessage = (event) => {
      try {
          const data = JSON.parse(event.data);
          console.log(`📩 Received from ${url}:`, data);
          onMessage(data);
      } catch (error) {
          console.error(`❌ WebSocket JSON Parse Error:`, error);
      }
  };

  socket.onerror = (error) => {
      console.error(`⚠️ WebSocket Error:`, error);
  };

  socket.onclose = () => {
      console.log(`🔴 Disconnected from WebSocket: ${url}`);
  };

  return socket;
};

export default connectWebSocket;

export const SOCKET_URLS = {
  sensorPredictions: 'ws://localhost:8000/ws/sensor_data_predictions',
  accidentPredictions: 'ws://127.0.0.1:8000/ws/accident_data_predictions'
};

export const createWebSocket = (url: string) => {
  const socket = new WebSocket(url);
  
  socket.onopen = () => {
    console.log(`Connected to WebSocket at ${url}`);
  };

  socket.onerror = (error) => {
    console.error(`WebSocket error for ${url}:`, error);
  };

  socket.onclose = () => {
    console.log(`Disconnected from WebSocket at ${url}`);
  };

  return socket;
};
