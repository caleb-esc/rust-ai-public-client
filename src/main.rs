use anyhow::{Context, Result};
use clap::Parser;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{self, Write};

#[derive(Parser, Debug)]
struct Cli {
    #[arg(
        help = "Prompt to send. If omitted, interactive chat mode is used.",
        value_name = "PROMPT"
    )]
    prompt: Option<String>,

    #[arg(
        long,
        default_value = "gemini",
        help = "AI provider to use: gemini or openai"
    )]
    provider: String,
}

#[derive(Debug, Clone, Copy)]
enum Provider {
    Gemini,
    Openai,
}

impl Provider {
    fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "gemini" | "google" => Ok(Self::Gemini),
            "openai" | "open_ai" => Ok(Self::Openai),
            _ => anyhow::bail!(
                "Unsupported provider '{}'. Use 'gemini' or 'openai'.",
                value
            ),
        }
    }
}

#[derive(Debug, Clone)]
struct AppConfig {
    provider: Provider,
    google_api_key: Option<String>,
    gemini_model: String,
    openai_api_key: Option<String>,
    openai_model: String,
}

fn load_config(provider_override: &str) -> Result<AppConfig> {
    dotenvy::dotenv().ok();

    let provider = Provider::parse(provider_override)?;

    let google_api_key = std::env::var("GOOGLE_API_KEY")
        .or_else(|_| std::env::var("GEMINI_API_KEY"))
        .ok();

    let openai_api_key = std::env::var("OPENAI_API_KEY").ok();

    let gemini_model = std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.0-flash".to_string());
    let openai_model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    Ok(AppConfig {
        provider,
        google_api_key,
        gemini_model,
        openai_api_key,
        openai_model,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = load_config(&cli.provider)?;

    let client = Client::new();

    match cli.prompt {
        Some(prompt) => {
            let response = generate_text(&client, &config, &prompt).await?;
            println!("{}", response);
            Ok(())
        }
        None => interactive_chat(&client, &config).await,
    }
}

async fn interactive_chat(client: &Client, config: &AppConfig) -> Result<()> {
    println!("AI chat started. Type 'exit' to quit.");
    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            continue;
        }

        if trimmed.eq_ignore_ascii_case("exit") || trimmed.eq_ignore_ascii_case("quit") {
            println!("Goodbye.");
            break;
        }

        match generate_text(client, config, trimmed).await {
            Ok(reply) => println!("AI: {}", reply),
            Err(err) => {
                eprintln!("Error: {}", err);
            }
        }
    }

    Ok(())
}

async fn generate_text(client: &Client, config: &AppConfig, prompt: &str) -> Result<String> {
    match config.provider {
        Provider::Gemini => call_gemini(client, config, prompt).await,
        Provider::Openai => call_openai(client, config, prompt).await,
    }
}

async fn call_gemini(client: &Client, config: &AppConfig, prompt: &str) -> Result<String> {
    let api_key = config
        .google_api_key
        .as_ref()
        .context("Missing GOOGLE_API_KEY or GEMINI_API_KEY in environment or .env")?;

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        config.gemini_model, api_key
    );

    let body = json!({
        "contents": [
            {
                "role": "user",
                "parts": [
                    { "text": prompt }
                ]
            }
        ]
    });

    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;

    let status = response.status();
    let text = response.text().await?;

    if !status.is_success() {
        anyhow::bail!("Gemini API error ({}): {}", status, text);
    }

    #[derive(Debug, Deserialize)]
    struct GeminiResponse {
        candidates: Vec<Candidate>,
    }

    #[derive(Debug, Deserialize)]
    struct Candidate {
        content: Content,
    }

    #[derive(Debug, Deserialize)]
    struct Content {
        parts: Vec<Part>,
    }

    #[derive(Debug, Deserialize)]
    struct Part {
        text: String,
    }

    let parsed: GeminiResponse = serde_json::from_str(&text)
        .context(format!("Failed to parse Gemini response: {}", text))?;

    let first = parsed
        .candidates
        .into_iter()
        .next()
        .context("Gemini API returned no candidates")?;

    let text = first
        .content
        .parts
        .into_iter()
        .map(|p| p.text)
        .collect::<Vec<_>>()
        .join("");

    Ok(text)
}

async fn call_openai(client: &Client, config: &AppConfig, prompt: &str) -> Result<String> {
    let api_key = config
        .openai_api_key
        .as_ref()
        .context("Missing OPENAI_API_KEY in environment or .env")?;

    let url = "https://api.openai.com/v1/chat/completions";

    let body = json!({
        "model": config.openai_model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7
    });

    let response = client
        .post(url)
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;

    let status = response.status();
    let text = response.text().await?;

    if !status.is_success() {
        anyhow::bail!("OpenAI API error ({}): {}", status, text);
    }

    #[derive(Debug, Deserialize)]
    struct OpenAIResponse {
        choices: Vec<Choice>,
    }

    #[derive(Debug, Deserialize)]
    struct Choice {
        message: Message,
    }

    #[derive(Debug, Deserialize)]
    struct Message {
        content: String,
    }

    let parsed: OpenAIResponse = serde_json::from_str(&text)
        .context(format!("Failed to parse OpenAI response: {}", text))?;

    let answer = parsed
        .choices
        .into_iter()
        .next()
        .context("OpenAI API returned no choices")?;

    Ok(answer.message.content)
}
