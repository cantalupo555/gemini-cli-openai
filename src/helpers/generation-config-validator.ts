import { geminiCliModels } from "../models";
import {
	DEFAULT_THINKING_BUDGET,
	DEFAULT_TEMPERATURE,
	REASONING_EFFORT_BUDGETS,
	GEMINI_SAFETY_CATEGORIES
} from "../constants";
import { ChatCompletionRequest, Env, EffortLevel, SafetyThreshold } from "../types";
import { NativeToolsConfiguration } from "../types/native-tools";

/**
 * Helper class to validate and correct generation configurations for different Gemini models.
 * Handles model-specific limitations and provides sensible defaults.
 */
export class GenerationConfigValidator {
	/**
	 * Maps reasoning effort to thinking budget based on model type.
	 * @param effort - The reasoning effort level
	 * @param modelId - The model ID to determine if it's a flash model
	 * @returns The corresponding thinking budget
	 */
	static mapEffortToThinkingBudget(effort: EffortLevel, modelId: string): number {
		const isFlashModel = modelId.includes("flash");

		switch (effort) {
			case "none":
				return REASONING_EFFORT_BUDGETS.none;
			case "low":
				return REASONING_EFFORT_BUDGETS.low;
			case "medium":
				return isFlashModel ? REASONING_EFFORT_BUDGETS.medium.flash : REASONING_EFFORT_BUDGETS.medium.default;
			case "high":
				return isFlashModel ? REASONING_EFFORT_BUDGETS.high.flash : REASONING_EFFORT_BUDGETS.high.default;
			default:
				return DEFAULT_THINKING_BUDGET;
		}
	}

	/**
	 * Type guard to check if a value is a valid EffortLevel.
	 * @param value - The value to check
	 * @returns True if the value is a valid EffortLevel
	 */
	static isValidEffortLevel(value: unknown): value is EffortLevel {
		return typeof value === "string" && ["none", "low", "medium", "high"].includes(value);
	}

	/**
	 * Creates safety settings configuration for Gemini API.
	 * @param env - Environment variables containing safety thresholds
	 * @returns Safety settings configuration
	 */
	static createSafetySettings(env: Env): Array<{ category: string; threshold: SafetyThreshold }> {
		const safetySettings: Array<{ category: string; threshold: SafetyThreshold }> = [];

		if (env.GEMINI_MODERATION_HARASSMENT_THRESHOLD) {
			safetySettings.push({
				category: GEMINI_SAFETY_CATEGORIES.HARASSMENT,
				threshold: env.GEMINI_MODERATION_HARASSMENT_THRESHOLD
			});
		}

		if (env.GEMINI_MODERATION_HATE_SPEECH_THRESHOLD) {
			safetySettings.push({
				category: GEMINI_SAFETY_CATEGORIES.HATE_SPEECH,
				threshold: env.GEMINI_MODERATION_HATE_SPEECH_THRESHOLD
			});
		}

		if (env.GEMINI_MODERATION_SEXUALLY_EXPLICIT_THRESHOLD) {
			safetySettings.push({
				category: GEMINI_SAFETY_CATEGORIES.SEXUALLY_EXPLICIT,
				threshold: env.GEMINI_MODERATION_SEXUALLY_EXPLICIT_THRESHOLD
			});
		}

		if (env.GEMINI_MODERATION_DANGEROUS_CONTENT_THRESHOLD) {
			safetySettings.push({
				category: GEMINI_SAFETY_CATEGORIES.DANGEROUS_CONTENT,
				threshold: env.GEMINI_MODERATION_DANGEROUS_CONTENT_THRESHOLD
			});
		}

		return safetySettings;
	}

	/**
	 * Validates and corrects the thinking budget for a specific model.
	 * @param modelId - The Gemini model ID
	 * @param thinkingBudget - The requested thinking budget
	 * @returns The corrected thinking budget
	 */
	static validateThinkingBudget(modelId: string, thinkingBudget: number): number {
		const modelInfo = geminiCliModels[modelId];

		// For thinking models, validate the budget
		if (modelInfo?.thinking) {
			// Gemini 2.5 Pro and Flash don't support thinking_budget: 0
			// They require -1 (dynamic allocation) or positive numbers
			if (thinkingBudget === 0) {
				console.log(`[GenerationConfig] Model '${modelId}' doesn't support thinking_budget: 0, using -1 instead`);
				return DEFAULT_THINKING_BUDGET; // -1
			}

			// Validate positive budget values (optional: add upper limits if needed)
			if (thinkingBudget < -1) {
				console.log(
					`[GenerationConfig] Invalid thinking_budget: ${thinkingBudget} for model '${modelId}', using -1 instead`
				);
				return DEFAULT_THINKING_BUDGET; // -1
			}
		}

		return thinkingBudget;
	}

	/**
	 * Creates a validated generation config for a specific model.
	 * @param modelId - The Gemini model ID
	 * @param options - Generation options including thinking budget and OpenAI parameters
	 * @param isRealThinkingEnabled - Whether real thinking is enabled
	 * @param includeReasoning - Whether to include reasoning in response
	 * @param env - Environment variables for safety settings
	 * @returns Validated generation configuration
	 */
	static createValidatedConfig(
		modelId: string,
		options: Partial<ChatCompletionRequest> = {},
		isRealThinkingEnabled: boolean,
		includeReasoning: boolean
	): Record<string, unknown> {
		const generationConfig: Record<string, unknown> = {
			temperature: options.temperature ?? DEFAULT_TEMPERATURE,
			maxOutputTokens: options.max_tokens,
			topP: options.top_p,
			stopSequences: typeof options.stop === "string" ? [options.stop] : options.stop,
			presencePenalty: options.presence_penalty,
			frequencyPenalty: options.frequency_penalty,
			seed: options.seed
		};

		if (options.response_format?.type === "json_object") {
			generationConfig.responseMimeType = "application/json";
		}

		const modelInfo = geminiCliModels[modelId];
		const isThinkingModel = modelInfo?.thinking || false;

		if (isThinkingModel) {
			let thinkingBudget = options.thinking_budget ?? DEFAULT_THINKING_BUDGET;

			// Handle reasoning effort mapping to thinking budget
			const reasoning_effort =
				options.reasoning_effort || options.extra_body?.reasoning_effort || options.model_params?.reasoning_effort;

			if (reasoning_effort && this.isValidEffortLevel(reasoning_effort)) {
				thinkingBudget = this.mapEffortToThinkingBudget(reasoning_effort, modelId);
				// If effort is "none", disable reasoning
				if (reasoning_effort === "none") {
					includeReasoning = false;
				} else {
					includeReasoning = true;
				}
			}

			const validatedBudget = this.validateThinkingBudget(modelId, thinkingBudget);

			if (isRealThinkingEnabled && includeReasoning) {
				// Enable thinking with validated budget
				generationConfig.thinkingConfig = {
					thinkingBudget: validatedBudget,
					includeThoughts: true // Critical: This enables thinking content in response
				};
				console.log(`[GenerationConfig] Real thinking enabled for '${modelId}' with budget: ${validatedBudget}`);
			} else {
				// For thinking models, always use validated budget (can't use 0)
				// Control thinking visibility with includeThoughts instead
				generationConfig.thinkingConfig = {
					thinkingBudget: this.validateThinkingBudget(modelId, DEFAULT_THINKING_BUDGET),
					includeThoughts: false // Disable thinking visibility in response
				};
			}
		}

		// Remove undefined keys
		Object.keys(generationConfig).forEach((key) => generationConfig[key] === undefined && delete generationConfig[key]);
		return generationConfig;
	}

	/**
	 * Filters a JSON Schema property to be compatible with Gemini API.
	 * Removes unsupported properties and converts complex types.
	 * @param property - The JSON Schema property to filter
	 * @returns A Gemini-compatible schema property
	 */
	private static filterSchemaProperty(property: unknown): unknown {
		if (typeof property !== "object" || property === null) {
			return property;
		}

		const prop = property as Record<string, unknown>;
		const filtered: Record<string, unknown> = {};

		// Handle 'type' - convert array of types to single type
		if (prop.type) {
			if (Array.isArray(prop.type)) {
				// If type is an array like ["string", "null"], pick the first non-null type
				const types = prop.type.filter((t: unknown) => t !== "null");
				filtered.type = types.length > 0 ? types[0] : "string";
			} else {
				filtered.type = prop.type;
			}
		}

		// Copy basic supported properties
		if (prop.description) filtered.description = prop.description;
		if (prop.enum) filtered.enum = prop.enum;
		if (prop.format) filtered.format = prop.format;

		// Handle nested properties recursively
		if (prop.properties && typeof prop.properties === "object") {
			filtered.properties = {};
			for (const [key, value] of Object.entries(prop.properties)) {
				(filtered.properties as Record<string, unknown>)[key] = this.filterSchemaProperty(value);
			}
		}

		// Handle array items recursively
		if (prop.items) {
			filtered.items = this.filterSchemaProperty(prop.items);
		}

		// Handle required fields
		if (Array.isArray(prop.required)) {
			filtered.required = prop.required;
		}

		// Handle additionalProperties
		if (prop.additionalProperties !== undefined) {
			if (typeof prop.additionalProperties === "boolean") {
				filtered.additionalProperties = prop.additionalProperties;
			} else {
				filtered.additionalProperties = this.filterSchemaProperty(prop.additionalProperties);
			}
		}

		// Handle minimum/maximum (but not exclusive variants)
		if (typeof prop.minimum === "number") filtered.minimum = prop.minimum;
		if (typeof prop.maximum === "number") filtered.maximum = prop.maximum;

		// Explicitly ignore unsupported properties:
		// - strict (OpenAI-specific)
		// - const (not supported by Gemini)
		// - $ref, $schema, $id (JSON Schema references)
		// - exclusiveMinimum, exclusiveMaximum (not supported)
		// - anyOf, oneOf, allOf, not (complex schemas)
		// - minLength, maxLength, pattern (may not be supported)

		return filtered;
	}

	/**
	 * Filters a complete JSON Schema to be compatible with Gemini API.
	 * @param schema - The JSON Schema to filter
	 * @returns A Gemini-compatible schema
	 */
	private static filterGeminiCompatibleSchema(schema: Record<string, unknown>): Record<string, unknown> {
		const filtered: Record<string, unknown> = {};

		// Copy type
		if (schema.type) {
			if (Array.isArray(schema.type)) {
				const types = schema.type.filter((t: unknown) => t !== "null");
				filtered.type = types.length > 0 ? types[0] : "object";
			} else {
				filtered.type = schema.type;
			}
		}

		// Copy description
		if (schema.description) {
			filtered.description = schema.description;
		}

		// Filter properties recursively
		if (schema.properties && typeof schema.properties === "object") {
			filtered.properties = {};
			for (const [key, value] of Object.entries(schema.properties)) {
				(filtered.properties as Record<string, unknown>)[key] = this.filterSchemaProperty(value);
			}
		}

		// Copy required fields
		if (Array.isArray(schema.required)) {
			filtered.required = schema.required;
		}

		// Handle additionalProperties
		if (schema.additionalProperties !== undefined) {
			if (typeof schema.additionalProperties === "boolean") {
				filtered.additionalProperties = schema.additionalProperties;
			} else {
				filtered.additionalProperties = this.filterSchemaProperty(schema.additionalProperties);
			}
		}

		return filtered;
	}

	static createValidateTools(options: Partial<ChatCompletionRequest> = {}) {
		const tools = [];
		let toolConfig = {};
		// Add tools configuration if provided
		if (Array.isArray(options.tools) && options.tools.length > 0) {
			const functionDeclarations = options.tools.map((tool) => {
				let parameters = tool.function.parameters;
				// Filter parameters for Gemini compatibility
				if (parameters) {
					try {
						parameters = this.filterGeminiCompatibleSchema(parameters);
						console.log(
							`[GenerationConfig] Filtered tool '${tool.function.name}' parameters for Gemini compatibility`
						);
					} catch (error) {
						console.error(`[GenerationConfig] Error filtering tool '${tool.function.name}':`, error);
						// Fallback to basic filtering if advanced filtering fails
						const before = parameters;
						parameters = Object.keys(parameters)
							.filter((key) => !key.startsWith("$"))
							.reduce(
								(after, key) => {
									after[key] = before[key];
									return after;
								},
								{} as Record<string, unknown>
							);
					}
				}
				return {
					name: tool.function.name,
					description: tool.function.description,
					parameters
				};
			});

			tools.push({ functionDeclarations });
			// Handle tool choice
			if (options.tool_choice) {
				if (options.tool_choice === "auto") {
					toolConfig = { functionCallingConfig: { mode: "AUTO" } };
				} else if (options.tool_choice === "none") {
					toolConfig = { functionCallingConfig: { mode: "NONE" } };
				} else if (typeof options.tool_choice === "object" && options.tool_choice.function) {
					toolConfig = {
						functionCallingConfig: {
							mode: "ANY",
							allowedFunctionNames: [options.tool_choice.function.name]
						}
					};
				}
			}
		}

		return { tools, toolConfig };
	}
	static createFinalToolConfiguration(
		config: NativeToolsConfiguration,
		options: Partial<ChatCompletionRequest> = {}
	): {
		tools: unknown[] | undefined;
		toolConfig: unknown | undefined;
	} {
		if (config.useCustomTools && config.customTools && config.customTools.length > 0) {
			// Use the filtered tools from createValidateTools instead of raw config.customTools
			const { tools: filteredTools, toolConfig } = this.createValidateTools(options);
			return {
				tools: filteredTools,
				toolConfig: toolConfig
			};
		}

		if (config.useNativeTools && config.nativeTools && config.nativeTools.length > 0) {
			return {
				tools: config.nativeTools.map((tool) => {
					if (tool.google_search) {
						return { google_search: tool.google_search };
					}
					if (tool.url_context) {
						return { url_context: tool.url_context };
					}
					return tool;
				}),
				toolConfig: undefined // Native tools don't use toolConfig in the same way
			};
		}

		// If no tools are enabled or the tool lists are empty, return undefined
		return { tools: undefined, toolConfig: undefined };
	}
}
