import axios from 'axios';
import FormData from 'form-data';
import { config } from '../config/env';
import { FlaskResponse } from '../types';

export const detectText = async (text: string): Promise<FlaskResponse> => {
  try {
    const response = await axios.post(`${config.FLASK_URL}/predict-text`, {
      text
    }, {
      headers: { 'Content-Type': 'application/json' },
      timeout: 30000
    });
    return response.data;
  } catch (error) {
    console.error('Flask text detection error:', error);
    throw new Error('Text detection service unavailable');
  }
};

export const detectImage = async (imageBuffer: Buffer, filename: string): Promise<FlaskResponse> => {
  try {
    const formData = new FormData();
    formData.append('image', imageBuffer, filename);

    const response = await axios.post(`${config.FLASK_URL}/predict-image`, formData, {
      headers: formData.getHeaders(),
      timeout: 30000
    });
    return response.data;
  } catch (error) {
    console.error('Flask image detection error:', error);
    throw new Error('Image detection service unavailable');
  }
};