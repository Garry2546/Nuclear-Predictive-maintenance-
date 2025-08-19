"use client";

import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { createWebSocket, SOCKET_URLS } from "@/utils/socket";

interface WebSocketContextType {
  sensorSocket: WebSocket | null;
  accidentSocket: WebSocket | null;
  sensorData: any;
  accidentData: any;
}

const WebSocketContext = createContext<WebSocketContextType>({
  sensorSocket: null,
  accidentSocket: null,
  sensorData: null,
  accidentData: null
});

export const useWebSocket = () => useContext(WebSocketContext);

export function WebSocketProvider({ children }: { children: ReactNode }) {
  const [sensorSocket, setSensorSocket] = useState<WebSocket | null>(null);
  const [accidentSocket, setAccidentSocket] = useState<WebSocket | null>(null);
  const [sensorData, setSensorData] = useState<any>(null);
  const [accidentData, setAccidentData] = useState<any>(null);

  useEffect(() => {
    const sensorWs = createWebSocket(SOCKET_URLS.sensorPredictions);
    const accidentWs = createWebSocket(SOCKET_URLS.accidentPredictions);

    sensorWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setSensorData(data);
      } catch (error) {
        console.error('Error parsing sensor data:', error);
      }
    };

    accidentWs.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setAccidentData(data);
      } catch (error) {
        console.error('Error parsing accident data:', error);
      }
    };

    setSensorSocket(sensorWs);
    setAccidentSocket(accidentWs);

    return () => {
      sensorWs.close();
      accidentWs.close();
    };
  }, []);

  return (
    <WebSocketContext.Provider value={{ 
      sensorSocket, 
      accidentSocket, 
      sensorData, 
      accidentData 
    }}>
      {children}
    </WebSocketContext.Provider>
  );
}