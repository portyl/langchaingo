package vertexaiclient

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"runtime"

	aiplatform "cloud.google.com/go/aiplatform/apiv1"
	"cloud.google.com/go/aiplatform/apiv1/aiplatformpb"
	"github.com/portyl/langchaingo/schema"
	"google.golang.org/api/option"
	"google.golang.org/protobuf/types/known/structpb"
)

const (
	defaultAPIEndpoint = "us-central1-aiplatform.googleapis.com:443"
	defaultLocation    = "us-central1"
	defaultPublisher   = "google"
)

var (
	// ErrMissingValue is returned when a value is missing.
	ErrMissingValue = errors.New("missing value")
	// ErrInvalidValue is returned when a value is invalid.
	ErrInvalidValue = errors.New("invalid value")
)

var defaultParameters = map[string]interface{}{ //nolint:gochecknoglobals
	"temperature":     0.2, //nolint:gomnd
	"maxOutputTokens": 256, //nolint:gomnd
	"topP":            0.8, //nolint:gomnd
	"topK":            40,  //nolint:gomnd
}

const (
	embeddingModelName = "textembedding-gecko"
	TextModelName      = "text-bison"
	ChatModelName      = "chat-bison"

	defaultMaxConns = 4
)

// PaLMClient represents a Vertex AI based PaLM API client.
type PaLMClient struct {
	client    *aiplatform.PredictionClient
	projectID string
	model     string
}

// New returns a new Vertex AI based PaLM API client.
func New(projectID string, opts ...option.ClientOption) (*PaLMClient, error) {
	numConns := runtime.GOMAXPROCS(0)
	if numConns > defaultMaxConns {
		numConns = defaultMaxConns
	}
	o := []option.ClientOption{
		option.WithGRPCConnectionPool(numConns),
		option.WithEndpoint(defaultAPIEndpoint),
	}
	o = append(o, opts...)

	ctx := context.Background()
	client, err := aiplatform.NewPredictionClient(ctx, o...)
	if err != nil {
		return nil, err
	}
	return &PaLMClient{
		client:    client,
		projectID: projectID,
	}, nil
}

// ErrEmptyResponse is returned when the OpenAI API returns an empty response.
var ErrEmptyResponse = errors.New("empty response")

// CompletionRequest is a request to create a completion.
type CompletionRequest struct {
	Prompts     []string `json:"prompts"`
	MaxTokens   int      `json:"max_tokens"`
	Temperature float64  `json:"temperature,omitempty"`
	TopP        int      `json:"top_p,omitempty"`
	TopK        int      `json:"top_k,omitempty"`
}

// Completion is a completion.
type Completion struct {
	Text string `json:"text"`
}

// CreateCompletion creates a completion.
func (c *PaLMClient) CreateCompletion(ctx context.Context, r *CompletionRequest) ([]*Completion, error) {
	params := map[string]interface{}{
		"maxOutputTokens": r.MaxTokens,
		"temperature":     r.Temperature,
		"top_p":           r.TopP,
		"top_k":           r.TopK,
	}
	predictions, err := c.batchPredict(ctx, TextModelName, r.Prompts, params)
	if err != nil {
		return nil, err
	}
	completions := []*Completion{}
	for _, p := range predictions {
		value := p.GetStructValue().AsMap()
		text, ok := value["content"].(string)
		if !ok {
			return nil, fmt.Errorf("%w: %v", ErrMissingValue, "content")
		}
		completions = append(completions, &Completion{
			Text: text,
		})
	}
	return completions, nil
}

// EmbeddingRequest is a request to create an embedding.
type EmbeddingRequest struct {
	Input []string `json:"input"`
}

// CreateEmbedding creates embeddings.
func (c *PaLMClient) CreateEmbedding(ctx context.Context, r *EmbeddingRequest) ([][]float64, error) {
	params := map[string]interface{}{}
	responses, err := c.batchPredict(ctx, embeddingModelName, r.Input, params)
	if err != nil {
		return nil, err
	}

	embeddings := [][]float64{}
	for _, res := range responses {
		value := res.GetStructValue().AsMap()
		embedding, ok := value["embeddings"].(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("%w: %v", ErrMissingValue, "embeddings")
		}
		values, ok := embedding["values"].([]interface{})
		if !ok {
			return nil, fmt.Errorf("%w: %v", ErrMissingValue, "values")
		}
		floatValues := []float64{}
		for _, v := range values {
			val, ok := v.(float64)
			if !ok {
				return nil, fmt.Errorf("%w: %v is not a float64", ErrInvalidValue, "value")
			}
			floatValues = append(floatValues, val)
		}
		embeddings = append(embeddings, floatValues)
	}
	return embeddings, nil
}

// ChatRequest is a request to create an embedding.
type ChatRequest struct {
	Context        string         `json:"context"`
	Messages       []*ChatMessage `json:"messages"`
	Temperature    float64        `json:"temperature,omitempty"`
	TopP           int            `json:"top_p,omitempty"`
	TopK           int            `json:"top_k,omitempty"`
	CandidateCount int            `json:"candidate_count,omitempty"`
	Model          string         `json:"-"`

	StreamingFunc func(context.Context, []byte) error
}

// ChatMessage is a message in a chat.
type ChatMessage struct {
	// The content of the message.
	Content string `json:"content"`
	// The name of the author of this message. user or bot
	Author string `json:"author,omitempty"`
}

type StreamedChatResponse = ChatResponse

// Statically assert that the types implement the interface.
var _ schema.ChatMessage = ChatMessage{}

// GetType returns the type of the message.
func (m ChatMessage) GetType() schema.ChatMessageType {
	switch m.Author {
	case "user":
		return schema.ChatMessageTypeHuman
	default:
		return schema.ChatMessageTypeAI
	}
}

// GetText returns the text of the message.
func (m ChatMessage) GetContent() string {
	return m.Content
}

// ChatResponse is a response to a chat request.
type ChatResponse struct {
	Candidates []ChatMessage `json:"candidates"`
}

// CreateChat creates chat request.
func (c *PaLMClient) CreateChat(ctx context.Context, r *ChatRequest) (*ChatResponse, error) {
	responses, stream, err := c.chat(ctx, r, r.StreamingFunc != nil)
	if err != nil {
		return nil, err
	}

	if stream != nil {
		defer stream.CloseSend()

		var msgAuthor string
		var msgContent string

		for stream.Context().Err() == nil {
			msg, err := stream.Recv()
			if err != nil {
				if err == io.EOF {
					break
				}

				return nil, err
			}

			if len(msg.Outputs) > 0 {
				result := convertTensorStructToMap(msg.Outputs[0])

				b, err := json.Marshal(result)
				if err != nil {
					return nil, err
				}

				var parsed StreamedChatResponse

				if err := json.Unmarshal(b, &parsed); err != nil {
					return nil, err
				}

				if len(parsed.Candidates) > 0 {
					c := parsed.Candidates[0]

					if msgAuthor == "" {
						msgAuthor = c.Author
					}

					msgContent += c.Content

					r.StreamingFunc(ctx, []byte(c.Content))
				}
			}
		}

		return &ChatResponse{
			Candidates: []ChatMessage{
				{Author: msgAuthor, Content: msgContent},
			},
		}, nil
	}

	chatResponse := &ChatResponse{}
	res := responses[0]
	value := res.GetStructValue().AsMap()
	candidates, ok := value["candidates"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("%w: %v", ErrMissingValue, "candidates")
	}
	for _, c := range candidates {
		candidate, ok := c.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("%w: %v is not a map[string]interface{}", ErrInvalidValue, "candidate")
		}
		author, ok := candidate["author"].(string)
		if !ok {
			return nil, fmt.Errorf("%w: %v is not a string", ErrInvalidValue, "author")
		}
		content, ok := candidate["content"].(string)
		if !ok {
			return nil, fmt.Errorf("%w: %v is not a string", ErrInvalidValue, "content")
		}
		chatResponse.Candidates = append(chatResponse.Candidates, ChatMessage{
			Author:  author,
			Content: content,
		})
	}
	return chatResponse, nil
}

func mergeParams(defaultParams, params map[string]interface{}) *structpb.Struct {
	mergedParams := map[string]interface{}{}
	for paramKey, paramValue := range defaultParameters {
		mergedParams[paramKey] = paramValue
	}
	for paramKey, paramValue := range params {
		switch value := paramValue.(type) {
		case float64:
			if value != 0 {
				mergedParams[paramKey] = value
			}
		case int:
		case int32:
		case int64:
			if value != 0 {
				mergedParams[paramKey] = value
			}
		}
	}
	smergedParams, err := structpb.NewStruct(mergedParams)
	if err != nil {
		smergedParams, _ = structpb.NewStruct(defaultParams)
		return smergedParams
	}
	return smergedParams
}

func (c *PaLMClient) batchPredict(ctx context.Context, model string, prompts []string, params map[string]interface{}) ([]*structpb.Value, error) { //nolint:lll
	mergedParams := mergeParams(defaultParameters, params)
	instances := []*structpb.Value{}
	for _, prompt := range prompts {
		content, _ := structpb.NewStruct(map[string]interface{}{
			"content": prompt,
		})
		instances = append(instances, structpb.NewStructValue(content))
	}
	resp, err := c.client.Predict(ctx, &aiplatformpb.PredictRequest{
		Endpoint:   c.projectLocationPublisherModelPath(c.projectID, "us-central1", "google", model),
		Instances:  instances,
		Parameters: structpb.NewStructValue(mergedParams),
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Predictions) == 0 {
		return nil, ErrEmptyResponse
	}
	return resp.Predictions, nil
}

func (c *PaLMClient) chat(
	ctx context.Context,
	r *ChatRequest,
	stream bool,
) ([]*structpb.Value, aiplatformpb.PredictionService_ServerStreamingPredictClient, error) {
	if c.model == "" {
		c.model = ChatModelName
	}

	if r.Model != "" {
		c.model = r.Model
	}

	params := map[string]interface{}{
		"temperature": r.Temperature,
		"top_p":       r.TopP,
		"top_k":       r.TopK,
	}
	mergedParams := mergeParams(defaultParameters, params)
	messages := []interface{}{}
	for _, msg := range r.Messages {
		msgMap := map[string]interface{}{
			"author":  msg.Author,
			"content": msg.Content,
		}
		messages = append(messages, msgMap)
	}
	instance := map[string]any{
		"context":  r.Context,
		"messages": messages,
	}
	instances := []map[string]any{instance}

	if stream {
		s, err := c.client.ServerStreamingPredict(ctx, &aiplatformpb.StreamingPredictRequest{
			Endpoint:   c.projectLocationPublisherModelPath(c.projectID, "us-central1", "google", c.model),
			Inputs:     convertToTensor(instances).ListVal,
			Parameters: convertToTensor(mergedParams),
		})
		if err != nil {
			return nil, nil, err
		}

		return nil, s, nil
	}

	formattedInstance, err := structpb.NewValue(instances)
	if err != nil {
		return nil, nil, err
	}
	formattedInstances := []*structpb.Value{formattedInstance}

	resp, err := c.client.Predict(ctx, &aiplatformpb.PredictRequest{
		Endpoint:   c.projectLocationPublisherModelPath(c.projectID, "us-central1", "google", c.model),
		Instances:  formattedInstances,
		Parameters: structpb.NewStructValue(mergedParams),
	})
	if err != nil {
		return nil, nil, err
	}
	if len(resp.Predictions) == 0 {
		return nil, nil, ErrEmptyResponse
	}
	return resp.Predictions, nil, nil
}

func (c *PaLMClient) projectLocationPublisherModelPath(projectID, location, publisher, model string) string {
	return fmt.Sprintf("projects/%s/locations/%s/publishers/%s/models/%s", projectID, location, publisher, model)
}

func convertToTensor(v interface{}) *aiplatformpb.Tensor {
	var tensor aiplatformpb.Tensor

	switch val := v.(type) {
	case string:
		tensor.Dtype = aiplatformpb.Tensor_STRING
		tensor.StringVal = append(tensor.StringVal, val)
	case bool:
		tensor.Dtype = aiplatformpb.Tensor_BOOL
		tensor.BoolVal = append(tensor.BoolVal, val)
	case int:
		tensor.Dtype = aiplatformpb.Tensor_INT64
		tensor.IntVal = append(tensor.IntVal, int32(val))
	case float64:
		tensor.Dtype = aiplatformpb.Tensor_FLOAT
		tensor.FloatVal = append(tensor.FloatVal, float32(val))
	case []interface{}:
		var converted []*aiplatformpb.Tensor

		for _, i := range val {
			converted = append(converted, convertToTensor(i))
		}

		tensor.ListVal = converted
	case []map[string]interface{}:
		var converted []*aiplatformpb.Tensor

		for _, i := range val {
			converted = append(converted, convertToTensor(i))
		}

		// tensor.Dtype = aiplatformpb.Tensor_DATA_TYPE_UNSPECIFIED
		tensor.ListVal = converted
	case map[string]interface{}:
		converted := make(map[string]*aiplatformpb.Tensor)

		for k, v := range val {
			converted[k] = convertToTensor(v)
		}

		// tensor.Dtype = aiplatformpb.Tensor_DATA_TYPE_UNSPECIFIED
		tensor.StructVal = converted
	default:
		log.Printf("%T\n", v)
	}

	return &tensor
}

func convertTensorStructToMap(tensor *aiplatformpb.Tensor) map[string]interface{} {
	result := make(map[string]interface{})

	if tensor == nil || tensor.StructVal == nil {
		return result
	}

	for key, value := range tensor.StructVal {
		switch {
		case len(value.StringVal) > 0:
			result[key] = value.StringVal[0]
		case len(value.BoolVal) > 0:
			result[key] = value.BoolVal[0]
		case len(value.Int64Val) > 0:
			result[key] = value.Int64Val[0]
		case len(value.FloatVal) > 0:
			result[key] = value.FloatVal[0]
		case len(value.ListVal) > 0: // assuming every item is a struct here
			list := make([]interface{}, len(value.ListVal))
			for i, item := range value.ListVal {
				list[i] = convertTensorStructToMap(item)
			}
			result[key] = list
		case value.StructVal != nil:
			result[key] = convertTensorStructToMap(value)
		}
	}

	return result

}
