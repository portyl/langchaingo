package cohereclient

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/cohere-ai/tokenizer"
)

var (
	ErrEmptyResponse = errors.New("empty response")
	ErrModelNotFound = errors.New("model not found")
)

type Client struct {
	token      string
	baseURL    string
	model      string
	httpClient Doer
	encoder    *tokenizer.Encoder
}

type Option func(*Client) error

// Doer performs a HTTP request.
type Doer interface {
	Do(req *http.Request) (*http.Response, error)
}

// WithHTTPClient allows setting a custom HTTP client.
func WithHTTPClient(client Doer) Option {
	return func(c *Client) error {
		c.httpClient = client

		return nil
	}
}

func New(token string, baseURL string, model string, opts ...Option) (*Client, error) {
	encoder, err := tokenizer.NewFromPrebuilt("coheretext-50k")
	if err != nil {
		return nil, fmt.Errorf("create tokenizer: %w", err)
	}

	c := &Client{
		token:      token,
		baseURL:    baseURL,
		model:      model,
		httpClient: http.DefaultClient,
		encoder:    encoder,
	}

	for _, opt := range opts {
		if err := opt(c); err != nil {
			return nil, err
		}
	}

	return c, nil
}

type GenerationRequest struct {
	Prompt string `json:"prompt"`

	// StreamingFunc is a function to be called for each chunk of a streaming response.
	// Return an error to stop streaming early.
	StreamingFunc func(ctx context.Context, chunk []byte) error `json:"-"`
}

type Generation struct {
	Text string `json:"text"`
}

type StreamedGeneration struct {
	Text         string `json:"text"`
	IsFinished   bool   `json:"is_finished"`
	FinishReason string `json:"finish_reason"`
	Response     struct {
		ID          string `json:"id"`
		Generations []struct {
			ID           string `json:"id"`
			Text         string `json:"text"`
			FinishReason string `json:"finish_reason"`
		} `json:"generations"`
		Prompt string `json:"prompt"`
	} `json:"response"`
}

type generateRequestPayload struct {
	Prompt string `json:"prompt"`
	Model  string `json:"model"`
	Stream bool   `json:"stream"`
}

type generateResponsePayload struct {
	ID          string `json:"id,omitempty"`
	Message     string `json:"message,omitempty"`
	Generations []struct {
		ID   string `json:"id,omitempty"`
		Text string `json:"text,omitempty"`
	} `json:"generations,omitempty"`
}

func (c *Client) CreateGeneration(ctx context.Context, r *GenerationRequest) (*Generation, error) {
	if c.baseURL == "" {
		c.baseURL = "https://api.cohere.ai"
	}

	payload := generateRequestPayload{
		Prompt: r.Prompt,
		Model:  c.model,
	}

	if r.StreamingFunc != nil {
		payload.Stream = true
	}

	payloadBytes, err := json.Marshal(&payload)
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		fmt.Sprintf("%s/v1/generate", c.baseURL),
		bytes.NewReader(payloadBytes),
	)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("content-type", "application/json")
	req.Header.Set("authorization", "bearer "+c.token)

	res, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}

	if r.StreamingFunc != nil {
		return parseStreamingChatResponse(ctx, res, r)
	}

	defer res.Body.Close()

	var response generateResponsePayload

	if err := json.NewDecoder(res.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	if len(response.Generations) == 0 {
		if strings.HasPrefix(response.Message, "model not found") {
			return nil, ErrModelNotFound
		}
		return nil, ErrEmptyResponse
	}

	var generation Generation
	generation.Text = response.Generations[0].Text

	return &generation, nil
}

func (c *Client) GetNumTokens(text string) int {
	encoded, _ := c.encoder.Encode(text)
	return len(encoded)
}

func parseStreamingChatResponse(ctx context.Context, r *http.Response, payload *GenerationRequest) (*Generation, error) { //nolint:cyclop,lll
	scanner := bufio.NewScanner(r.Body)
	responseChan := make(chan StreamedGeneration)
	go func() {
		defer close(responseChan)
		for scanner.Scan() {
			line := scanner.Text()
			if line == "" {
				continue
			}
			data := strings.Trim(line, "\r\t\n")
			var streamPayload StreamedGeneration
			err := json.NewDecoder(bytes.NewReader([]byte(data))).Decode(&streamPayload)
			if err != nil {
				log.Fatalf("failed to decode stream payload: %v", err)
			}
			responseChan <- streamPayload
			if streamPayload.IsFinished {
				return
			}
		}
		if err := scanner.Err(); err != nil {
			log.Println("issue scanning response:", err)
		}
	}()
	// Parse response
	response := Generation{}

	for streamResponse := range responseChan {
		chunk := []byte(streamResponse.Text)
		if payload.StreamingFunc != nil {
			err := payload.StreamingFunc(ctx, chunk)
			if err != nil {
				return nil, fmt.Errorf("streaming func returned an error: %w", err)
			}
		}
	}
	return &response, nil
}
