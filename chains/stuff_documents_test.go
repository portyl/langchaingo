package chains

import (
	"context"
	"os"
	"testing"

	"github.com/portyl/langchaingo/llms/openai"
	"github.com/portyl/langchaingo/prompts"
	"github.com/portyl/langchaingo/schema"
	"github.com/stretchr/testify/require"
)

func TestStuffDocuments(t *testing.T) {
	t.Parallel()

	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	model, err := openai.New()
	require.NoError(t, err)

	prompt := prompts.NewPromptTemplate(
		"Write {{.context}}",
		[]string{"context"},
	)
	require.NoError(t, err)

	llmChain := NewLLMChain(model, prompt)
	chain := NewStuffDocuments(llmChain)

	docs := []schema.Document{
		{PageContent: "foo"},
		{PageContent: "bar"},
		{PageContent: "baz"},
	}

	result, err := Call(context.Background(), chain, map[string]any{
		"input_documents": docs,
	})
	require.NoError(t, err)
	for _, key := range chain.GetOutputKeys() {
		_, ok := result[key]
		require.True(t, ok)
	}
}
