
import "dotenv/config";
import {LlmAgent, LlmAgentConfig} from '@google/adk';

const llmAgentConfig: LlmAgentConfig = {
  name: "my-llm-agent",
  model: "gemini-2.5-flash",
};
const agent = new LlmAgent(llmAgentConfig);

export default agent;