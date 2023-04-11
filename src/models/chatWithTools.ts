import { AgentExecutor, Tool, initializeAgentExecutor } from "langchain/agents";
import { ChatOpenAI } from "langchain/chat_models";
import { BufferMemory } from "langchain/memory";
import { Configuration } from "openai";
import { OpenAIApi } from "openai";
import { googleTool } from "./tools/google";

const openAIApiKey = process.env.OPENAI_API_KEY!;
const model = process.env.OPENAI_MODEL || "gpt-4";

const params = {
  verbose: true,
  temperature: 1,
  openAIApiKey,
  modelName: model,
  maxConcurrency: 1,
  maxTokens: model === 'gpt-4' ? 4000 : 2000,
  maxRetries: 5,
};

export class Model {
  public tools: Tool[];
  public executor?: AgentExecutor;
  public openai: OpenAIApi;
  public model: ChatOpenAI;

  constructor() {
    const configuration = new Configuration({
      apiKey: openAIApiKey,
    });

    this.tools = [googleTool];
    this.openai = new OpenAIApi(configuration);
    this.model = new ChatOpenAI(params, configuration);
  }

  public async call(input: string) {
    if (!this.executor) {
      this.executor = await initializeAgentExecutor(
        this.tools,
        this.model,
        "chat-conversational-react-description",
        true
      );
      this.executor.memory = new BufferMemory({
        returnMessages: true,
        memoryKey: "chat_history",
        inputKey: "input",
      });
    }

    const response = await this.executor!.call({ input });

    console.log("Model response: " + response);

    return response.output;
  }
}
