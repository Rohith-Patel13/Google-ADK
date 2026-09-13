import "dotenv/config";

import {
  Pinecone,
  PineconeRecord,
  RecordMetadata,
} from "@pinecone-database/pinecone";

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!,
});

// Must match the `outputDimensionality` used when generating embeddings
// with geminiClientLibrary.generateEmbedding(), otherwise upserts/queries
// will fail with a dimension mismatch.
const EMBEDDING_DIMENSION = 768;

class PineconeClientLibrary {
  async createIndex(name: string) {
    try {
      const index = await pc.createIndex({
        name,
        dimension: EMBEDDING_DIMENSION,
        metric: "cosine",
        spec: {
          serverless: {
            cloud: "aws", // or "gcp" / "azure" — match what you picked in the console
            region: "us-east-1", // match your selected region
          },
        },
        vectorType: "dense",
        // Don't throw if the index already exists, and wait for the
        // serverless index to be ready before the caller starts upserting.
        suppressConflicts: true,
        waitUntilReady: true,
      });
      console.log(`Index '${name}' created successfully.`);
      return index;
    } catch (error) {
      console.error(`Error creating index '${name}':`, error);
      throw error;
    }
  }

  async addDocument(
    indexName: string,
    records: PineconeRecord<RecordMetadata>[],
  ) {
    try {
      const index = pc.index({ name: indexName });
      await index.upsert({ records });
      console.log(
        `${records.length} document(s) added to index '${indexName}' successfully.`,
      );
    } catch (error) {
      console.error(`Error adding document to index '${indexName}':`, error);
      throw error;
    }
  }

  async vectorSearch(indexName: string, vector: number[], topK: number = 5) {
    try {
      const index = pc.index({ name: indexName });
      const results = await index.query({
        vector,
        topK,
        includeMetadata: true,
      });
      console.log(`Vector search completed in index '${indexName}'.`);
      return results;
    } catch (error) {
      console.error(
        `Error performing vector search in index '${indexName}':`,
        error,
      );
      throw error;
    }
  }
}

const pineconeClientLibrary = new PineconeClientLibrary();
export { pineconeClientLibrary, EMBEDDING_DIMENSION };
