import { GoogleGenAI, Type } from "@google/genai";
import { KeywordAnalysis } from '../types';

// This is a placeholder; in a real environment, the API key is set securely.
const apiKey = process.env.API_KEY;
if (!apiKey) {
  console.warn("API_KEY environment variable not set. Gemini API calls will fail.");
}

const ai = new GoogleGenAI({ apiKey });

export const analyzeArticleKeywords = async (title: string, summary: string): Promise<KeywordAnalysis[]> => {
  const prompt = `
    Analyze the following academic paper's title and summary to identify the top 10 most relevant and important 
    technical terms, topics, and concepts. Count the occurrences of each key term (including its synonyms and 
    related forms) within the combined text.
    Return a JSON array of objects, where each object contains a "keyword" (the normalized term) and its "frequency".
    Sort the final list by frequency in descending order.

    Title: "${title}"

    Summary: "${summary}"
  `;

  const responseSchema = {
    type: Type.ARRAY,
    items: {
      type: Type.OBJECT,
      properties: {
        keyword: {
          type: Type.STRING,
          description: 'A key technical term or topic found in the text.',
        },
        frequency: {
          type: Type.NUMBER,
          description: 'The number of times the keyword or its variants appeared.',
        },
      },
      required: ['keyword', 'frequency'],
    },
  };

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: responseSchema,
      },
    });

    const jsonText = response.text.trim();
    const result = JSON.parse(jsonText);
    return result as KeywordAnalysis[];

  } catch (error) {
    console.error("Gemini API call failed:", error);
    // Provide a more user-friendly error message
    if (error instanceof Error && error.message.includes('API key')) {
         throw new Error("Failed to analyze keywords: The API key is invalid or missing. Please check your configuration.");
    }
    throw new Error("Failed to analyze article keywords. The AI model may be temporarily unavailable.");
  }
};