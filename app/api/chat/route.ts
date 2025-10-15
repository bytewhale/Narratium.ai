import { NextRequest, NextResponse } from "next/server";
import { ChatOpenAI } from "@langchain/openai";
import { ChatOllama } from "@langchain/ollama";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { llmType, baseUrl, model, apiKey } = body;

    if (!llmType || !baseUrl || !model) {
      return NextResponse.json(
        { success: false, error: "Missing required parameters" },
        { status: 400 },
      );
    }

    // For Ollama on Windows, ensure proper URL formatting
    let finalBaseUrl = baseUrl;
    if (llmType === "ollama") {
      // Handle Windows-specific URL issues
      if (finalBaseUrl === "localhost:11434" || finalBaseUrl === "11434") {
        finalBaseUrl = "http://localhost:11434";
      } else if (finalBaseUrl.startsWith("localhost:") && !finalBaseUrl.startsWith("http://")) {
        finalBaseUrl = "http://" + finalBaseUrl;
      } else if (!finalBaseUrl.startsWith("http://") && !finalBaseUrl.startsWith("https://")) {
        finalBaseUrl = "http://" + finalBaseUrl;
      }

      // Remove trailing slash if present
      if (finalBaseUrl.endsWith("/")) {
        finalBaseUrl = finalBaseUrl.slice(0, -1);
      }
    }

    // Initialize the appropriate LangChain client based on LLM type
    const chatModel = llmType === "openai"
      ? new ChatOpenAI({
        modelName: model,
        openAIApiKey: apiKey,
        configuration: {
          baseURL: baseUrl,
        },
        timeout: 30000, // 30 second timeout
      })
      : new ChatOllama({
        baseUrl: finalBaseUrl,
        model: model,
        temperature: 0.1, // Lower temperature for more consistent test responses
      });

    // Send test message using LangChain with simpler format for better compatibility
    const testMessage = llmType === "ollama"
      ? "Hi"  // Very simple message for Ollama to avoid prompt issues
      : "Hello, this is a test message. Please respond with 'Test successful' if you can read this.";

    const messages = llmType === "ollama"
      ? [{ role: "user", content: testMessage }]
      : [
        { role: "system", content: "You are a helpful AI assistant." },
        { role: "user", content: testMessage },
      ];
    console.log("messages", messages);
    const response = await chatModel.invoke(messages);
    console.log("response", response);
    const responseContent = response.content.toString().trim();

    // More flexible response validation - just check if we got any meaningful response
    if (responseContent && responseContent.length > 0) {
      return NextResponse.json({
        success: true,
        message: "Model test successful",
        response: responseContent,
      });
    } else {
      return NextResponse.json(
        { success: false, error: "Empty or invalid response from model" },
        { status: 500 },
      );
    }
  } catch (error) {
    console.error("Model test error:", error);
    return NextResponse.json(
      { success: false, error: error.message || "Failed to test model" },
      { status: 500 },
    );
  }
}
