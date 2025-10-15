import { NextRequest, NextResponse } from "next/server";
import { ChatOpenAI } from "@langchain/openai";
import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnablePassthrough } from "@langchain/core/runnables";

// 定义 LLM 配置接口
interface LLMConfig {
  modelName: string;
  apiKey: string;
  baseUrl?: string;
  llmType: "openai" | "ollama";
  temperature?: number;
  maxTokens?: number;
  maxRetries?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
  topK?: number;
  repeatPenalty?: number;
  streaming?: boolean;
  streamUsage?: boolean;
  language?: "zh" | "en";
}

// 创建 LLM 实例
function createLLM(config: LLMConfig): ChatOpenAI | ChatOllama {
  const safeModel = config.modelName?.trim() || "";
  const defaultSettings = {
    temperature: 0.7,
    maxTokens: undefined,
    timeout: 1000000000,
    maxRetries: 0,
    topP: 0.7,
    frequencyPenalty: 0,
    presencePenalty: 0,
    topK: 40,
    repeatPenalty: 1.1,
    streaming: false,
    streamUsage: true, // 默认启用token usage追踪
  };

  if (config.llmType === "openai") {
    return new ChatOpenAI({
      modelName: safeModel,
      openAIApiKey: config.apiKey,
      configuration: {
        baseURL: config.baseUrl?.trim() || undefined,
      },
      temperature: config.temperature ?? defaultSettings.temperature,
      maxRetries: config.maxRetries ?? defaultSettings.maxRetries,
      topP: config.topP ?? defaultSettings.topP,
      frequencyPenalty: config.frequencyPenalty ?? defaultSettings.frequencyPenalty,
      presencePenalty: config.presencePenalty ?? defaultSettings.presencePenalty,
      streaming: config.streaming ?? defaultSettings.streaming,
      streamUsage: config.streamUsage ?? defaultSettings.streamUsage,
    });
  } else if (config.llmType === "ollama") {
    return new ChatOllama({
      model: safeModel,
      baseUrl: config.baseUrl?.trim() || "http://localhost:11434",
      temperature: config.temperature ?? defaultSettings.temperature,
      topK: config.topK ?? defaultSettings.topK,
      topP: config.topP ?? defaultSettings.topP,
      frequencyPenalty: config.frequencyPenalty ?? defaultSettings.frequencyPenalty,
      presencePenalty: config.presencePenalty ?? defaultSettings.presencePenalty,
      repeatPenalty: config.repeatPenalty ?? defaultSettings.repeatPenalty,
      streaming: config.streaming ?? defaultSettings.streaming,
    });
  } else {
    throw new Error(`Unsupported LLM type: ${config.llmType}`);
  }
}

// 创建对话链
function createDialogueChain(llm: ChatOpenAI | ChatOllama): any {
  const dialoguePrompt = ChatPromptTemplate.fromMessages([
    ["system", "{system_message}"],
    ["human", "{user_message}"],
  ]);

  return RunnablePassthrough.assign({
    system_message: (input: any) => input.system_message,
    user_message: (input: any) => input.user_message,
  })
    .pipe(dialoguePrompt)
    .pipe(llm)
    .pipe(new StringOutputParser());
}

// 主要的 LLM 调用函数
async function invokeLLM(
  systemMessage: string,
  userMessage: string,
  config: LLMConfig,
): Promise<{ response: string; tokenUsage?: any }> {
  try {
    console.log("invokeLLM");

    // 为了获取真实的token usage，我们需要直接调用LLM而不是使用chain
    if (config.llmType === "openai") {
      const openaiLlm = createLLM(config) as ChatOpenAI;

      // 直接调用LLM获取完整的AIMessage响应
      const aiMessage = await openaiLlm.invoke([
        { role: "system", content: systemMessage },
        { role: "user", content: userMessage },
      ]);

      // 提取token usage信息
      let tokenUsage = null;
      if (aiMessage.usage_metadata) {
        tokenUsage = {
          prompt_tokens: aiMessage.usage_metadata.input_tokens,
          completion_tokens: aiMessage.usage_metadata.output_tokens,
          total_tokens: aiMessage.usage_metadata.total_tokens,
        };
      } else if (aiMessage.response_metadata?.tokenUsage) {
        // 兼容旧版本格式
        tokenUsage = aiMessage.response_metadata.tokenUsage;
      } else if (aiMessage.response_metadata?.usage) {
        // 兼容另一种格式
        tokenUsage = aiMessage.response_metadata.usage;
      }

      // 如果没有从响应中获取到token usage，尝试从流式响应中获取
      if (!tokenUsage && config.streaming && config.streamUsage) {
        console.log("📊 Token usage not found in response, this may be due to streaming mode");
      }

      return {
        response: aiMessage.content as string,
        tokenUsage,
      };
    } else {
      // 对于其他LLM类型，使用原来的chain方式
      const llm = createLLM(config);
      const dialogueChain = createDialogueChain(llm);
      const response = await dialogueChain.invoke({
        system_message: systemMessage,
        user_message: userMessage,
      });

      if (!response || typeof response !== "string") {
        throw new Error("Invalid response from LLM");
      }

      return { response };
    }
  } catch (error) {
    console.error("Error in invokeLLM:", error);
    throw error;
  }
}

// POST 请求处理函数
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { systemMessage, userMessage, config } = body;

    // 验证必要参数
    if (!systemMessage || !userMessage || !config) {
      return NextResponse.json(
        { success: false, error: "Missing required parameters" },
        { status: 400 },
      );
    }

    // 调用 LLM
    const result = await invokeLLM(systemMessage, userMessage, config);

    // 返回结果
    return NextResponse.json({
      success: true,
      response: result.response,
      tokenUsage: result.tokenUsage,
    });
  } catch (error: any) {
    console.error("LLM API error:", error);
    return NextResponse.json(
      { success: false, error: error.message || "Failed to process LLM request" },
      { status: 500 },
    );
  }
}
