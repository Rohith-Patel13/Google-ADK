import "dotenv/config";
import { FunctionTool, LlmAgent, LlmAgentConfig } from "@google/adk";
import { z } from "zod";
import { geminiClientLibrary } from "./libraries/gemini-client.library";
import {
  pineconeClientLibrary,
  EMBEDDING_DIMENSION,
} from "./libraries/pinecone-client.library";

// The Pinecone index that holds the knowledge-base documents
// (students, university, clubs, etc.). Records store their text under
// the `document` metadata field.
const KNOWLEDGE_INDEX = "test-index";

/**
 * RAG retrieval tool.
 *
 * The LLM decides *whether* to call this. When a question needs a fact from
 * the knowledge base, the model calls it with a natural-language `query`;
 * we embed that query with Gemini and run a Pinecone vector search, then hand
 * the matching documents back to the model to answer from.
 */
const searchKnowledge = new FunctionTool({
  name: "search_knowledge",
  description:
    "Search the knowledge base for facts about students, people, clubs, or " +
    "the university (e.g. someone's age, details, or memberships). Use this " +
    "whenever the user asks about a specific person or detail that would be " +
    "stored in our records. Do NOT use it for greetings or general knowledge.",
  parameters: z.object({
    query: z
      .string()
      .describe(
        "A concise natural-language description of the information to look up, " +
          "e.g. 'age of Alexandra Thompson' or 'university chess club'.",
      ),
  }),
  execute: async ({ query }) => {
    // 1) Turn the text query into an embedding vector (must match the
    //    dimension the documents were indexed with).
    const embedding = await geminiClientLibrary.generateEmbedding(query, {
      taskType: "RETRIEVAL_QUERY",
      outputDimensionality: EMBEDDING_DIMENSION,
    });

    const vector = embedding.embeddings?.[0]?.values;
    if (!vector) {
      return { results: [], error: `Failed to embed query: "${query}"` };
    }

    // 2) Semantic search in Pinecone.
    const search = await pineconeClientLibrary.vectorSearch(
      KNOWLEDGE_INDEX,
      vector,
    );

    // 3) Return clean, model-friendly context (score + the document text).
    const results = (search.matches ?? []).map((m) => ({
      score: m.score,
      document: m.metadata?.document ?? m.metadata,
    }));

    return { results };
  },
});

const llmAgentConfig: LlmAgentConfig = {
  name: "my-llm-agent",
  model: "gemini-2.5-flash",
  instruction:
    "You are a helpful assistant with access to a private knowledge base " +
    "(students, people, clubs, and university details).\n" +
    "- When the user asks about a specific person or fact that would live in " +
    "our records (for example someone's age or details), call the " +
    "`search_knowledge` tool and answer STRICTLY from the documents it returns.\n" +
    "- If the search returns nothing relevant, say you don't have that " +
    "information — do not guess.\n" +
    "- For greetings, small talk, or general knowledge, just answer directly " +
    "without calling the tool.",
  tools: [searchKnowledge],
};

const agent = new LlmAgent(llmAgentConfig);

export default agent;
