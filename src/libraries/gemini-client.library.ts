import "dotenv/config";
import { EmbedContentResponse, GoogleGenAI } from "@google/genai";

class GeminiClientLibrary {
  ai = new GoogleGenAI({
    // The ADK project's .env exposes the key as GOOGLE_GENAI_API_KEY
    // (the same key ADK itself uses to talk to the Gemini backend).
    apiKey: process.env.GOOGLE_GENAI_API_KEY!,
  });

  async generateEmbedding(
    input: string,
    config: {
      taskType?: "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY";
      outputDimensionality?: number;
    } = {},
  ) {
    const response: EmbedContentResponse = await this.ai.models.embedContent({
      model: "gemini-embedding-001",
      contents: input,
      config,
    });

    return response;
  }
}

const geminiClientLibrary = new GeminiClientLibrary();
export { geminiClientLibrary };
