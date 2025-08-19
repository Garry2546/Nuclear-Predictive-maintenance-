import api from '@/utils/api';

// Types for your API responses
export interface ChecklistItem {
  id: string;
  text: string;
  checked: boolean;
}

export interface ChecklistSection {
  id: string;
  title: string;
  items: ChecklistItem[];
  isExpanded: boolean;
}

export interface SystemStatus {
  id: string;
  name: string;
  status: 'stable' | 'warning' | 'error';
  metrics: Record<string, number | string>;
}

export interface Summary {
  title?: string;
  sections?: {
    title: string;
    items: string[];
  }[];
}

export interface PumpData {
  time: string;
  classification: 'NORMAL' | 'RECOVERING' | 'BROKEN';
}

export interface EmergencyAlert {
  time: string;
  message: string;
}

interface WebSocketOptions {
  onData: (data: any) => void;
  onError?: (error: string) => void;
  onConnectionChange?: (connected: boolean) => void;
}

class WebSocketManager {
  private socket: WebSocket | null = null;
  private url: string;
  private options: WebSocketOptions;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private isConnecting = false;

  constructor(url: string, options: WebSocketOptions) {
    this.url = url;
    this.options = options;
    this.connect();
  }

  private async connect() {
    if (this.isConnecting) return;
    this.isConnecting = true;

    try {
      console.log(`Attempting to connect to ${this.url}`);

      if (this.socket) {
        this.socket.close();
      }

      this.socket = new WebSocket(this.url);

      // Set a connection timeout
      const connectionTimeout = setTimeout(() => {
        if (this.socket?.readyState !== WebSocket.OPEN) {
          this.socket?.close();
          throw new Error('Connection timeout after 5 seconds');
        }
      }, 5000);

      this.socket.onopen = () => {
        console.log(`Connected successfully to ${this.url}`);
        clearTimeout(connectionTimeout);
        this.reconnectAttempts = 0;
        this.isConnecting = false;
        this.options.onConnectionChange?.(true);
      };

      this.socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('Received data:', data);
          this.options.onData(data);
        } catch (error) {
          console.error('Error parsing message:', error);
          this.options.onError?.('Failed to parse server message');
        }
      };

      this.socket.onclose = (event) => {
        console.log(`WebSocket closed:`, {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean
        });
        clearTimeout(connectionTimeout);
        this.isConnecting = false;
        this.options.onConnectionChange?.(false);
        
        if (!event.wasClean) {
          this.reconnect();
        }
      };

      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        clearTimeout(connectionTimeout);
        this.isConnecting = false;
        this.options.onError?.('Failed to establish WebSocket connection');
      };

    } catch (error) {
      console.error('Connection error:', error);
      this.isConnecting = false;
      this.options.onError?.(error instanceof Error ? error.message : 'Failed to connect');
      this.reconnect();
    }
  }

  private reconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }

    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Reconnecting... Attempt ${this.reconnectAttempts}`);
      this.reconnectTimer = setTimeout(() => this.connect(), 3000);
    } else {
      this.options.onError?.(`Failed to connect after ${this.maxReconnectAttempts} attempts. Please check if the server is running.`);
    }
  }

  public disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }

  public getStatus() {
    return {
      connected: this.socket?.readyState === WebSocket.OPEN,
      readyState: this.socket?.readyState,
      reconnectAttempts: this.reconnectAttempts,
      url: this.url
    };
  }
}

// API methods
export const apiService = {
  // Checklist endpoints
  getChecklist: async (): Promise<ChecklistSection[]> => {
    try {
      const response = await api.get('/checklist-data');
      const data = response.data.checklist;
      
      const sections: ChecklistSection[] = [];
      let currentSection: ChecklistSection | null = null;
      let totalItems = 0;
      const ITEM_LIMIT = 20;
      let sectionCounter = 0;

      // Process each line
      for (const line of data) {
        // Stop processing if we've reached 20 items
        if (totalItems > ITEM_LIMIT) {
          if (currentSection) {
            sections.push(currentSection);
          }
          break;
        }

        if (line.startsWith('## ')) {
          // If we were building a previous section, add it to our sections array
          if (currentSection && currentSection.items.length > 0) {
            sections.push(currentSection);
          }
          
          // Increment section counter
          sectionCounter++;
          const uniqueSectionId = `section-${Date.now()}-${sectionCounter}`;
          
          // Start a new section
          currentSection = {
            id: uniqueSectionId,
            title: line.replace('## ', '').replace('\\n', ''),
            items: [],
            isExpanded: true
          };
        } else if (line.startsWith('- ') && currentSection) {
          // Only add item if we haven't reached the limit
          if (totalItems < ITEM_LIMIT) {
            const uniqueItemId = `${currentSection.id}-item-${Date.now()}-${currentSection.items.length + 1}`;
            currentSection.items.push({
              id: uniqueItemId,
              text: line.replace('- ', '').replace('\\n', ''),
              checked: false
            });
            totalItems++;
          }
        }
      }

      // Add the last section if it has items and we haven't reached the limit
      if (currentSection && currentSection.items.length > 0 && totalItems <= ITEM_LIMIT) {
        sections.push(currentSection);
      }

      // Filter out any sections without items
      const filteredSections = sections.filter(section => section.items.length > 0);

      return filteredSections;
    } catch (error) {
      console.error('Error fetching checklist:', error);
      throw error;
    }
  },

  updateChecklistItem: async (sectionId: string, itemId: string, checked: boolean) => {
    // Since we don't have a backend endpoint for updates yet,
    // we'll just return a success response
    return { success: true };
  },

  // System status endpoints
  getSystemStatus: async () => {
    const response = await api.get<SystemStatus[]>('/system-status');
    return response.data;
  },

  // Add more API methods as needed
  getReactorStatus: async () => {
    const response = await api.get('/reactor-status');
    return response.data;
  },

  getPumpStatus: async () => {
    const response = await api.get('/pump-status');
    return response.data;
  },

  getTurbineStatus: async () => {
    const response = await api.get('/turbine-status');
    return response.data;
  },

  // Add new method for summary
  getSummary: async (): Promise<Summary> => {
    try {
      const response = await api.get('/summary-extractor');
      console.log('Raw API Response:', response.data);
      
      // Get the summary array from the response
      const summaryArray = response.data.summary;
      
      if (!Array.isArray(summaryArray)) {
        throw new Error('Invalid data format received from API');
      }

      // Initialize the formatted structure
      const formattedSummary: Summary = {
        title: '',
        sections: []
      };

      let currentSection: { title: string; items: string[] } | null = null;

      // Process each line in the array
      summaryArray.forEach((line: string) => {
        // Clean the line
        line = line.trim().replace(/\\n$/, '');
        
        if (!line) return; // Skip empty lines
        
        // Main title (first occurrence of **)
        if (!formattedSummary.title && line.startsWith('**') && line.endsWith('**')) {
          formattedSummary.title = line.replace(/\*\*/g, '').trim();
          return;
        }
        
        // Section title (starts with ** but not the main title)
        if (line.startsWith('**') && line.endsWith('**')) {
          if (currentSection) {
            formattedSummary.sections?.push(currentSection);
          }
          currentSection = {
            title: line.replace(/\*\*/g, '').trim(),
            items: []
          };
        }
        // List item (starts with -)
        else if (line.startsWith('-') && currentSection) {
          const itemText = line.replace(/^-\s*/, '').trim();
          currentSection.items.push(itemText);
        }
      });

      // Add the last section
      if (currentSection) {
        formattedSummary.sections?.push(currentSection);
      }

      console.log('Formatted Summary:', formattedSummary);
      return formattedSummary;
    } catch (error) {
      console.error('Error fetching summary:', error);
      throw error;
    }
  },

  // Updated WebSocket methods with the typical FastAPI WebSocket URL pattern
  subscribeToPumpData: (
    onData: (data: PumpData) => void,
    onError?: (error: string) => void,
    onConnectionChange?: (connected: boolean) => void
  ) => {
    // Using the standard FastAPI WebSocket URL pattern
    const wsUrl = `ws://${window.location.hostname}:8000/ws/sensor_data_predictions/`;
    console.log('Connecting to pump data WebSocket:', wsUrl);
    
    return new WebSocketManager(
      wsUrl,
      { onData, onError, onConnectionChange }
    );
  },

  subscribeToEmergencyAlerts: (
    onData: (data: EmergencyAlert) => void,
    onError?: (error: string) => void,
    onConnectionChange?: (connected: boolean) => void
  ) => {
    const wsUrl = `ws://${window.location.hostname}:8000/ws/accident_data_predictions/`;
    console.log('Connecting to emergency alerts WebSocket:', wsUrl);
    
    return new WebSocketManager(
      wsUrl,
      { onData, onError, onConnectionChange }
    );
  },
};