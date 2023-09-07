package agents_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/portyl/langchaingo/agents"
	"github.com/portyl/langchaingo/chains"
	"github.com/portyl/langchaingo/llms/openai"
	"github.com/portyl/langchaingo/tools"
	"github.com/portyl/langchaingo/tools/serpapi"
	"github.com/stretchr/testify/require"
)

func TestMRKL(t *testing.T) {
	t.Parallel()

	if openaiKey := os.Getenv("OPENAI_API_KEY"); openaiKey == "" {
		t.Skip("OPENAI_API_KEY not set")
	}
	if serpapiKey := os.Getenv("SERPAPI_API_KEY"); serpapiKey == "" {
		t.Skip("SERPAPI_API_KEY not set")
	}

	llm, err := openai.New()
	require.NoError(t, err)

	searchTool, err := serpapi.New()
	require.NoError(t, err)

	calculator := tools.Calculator{}

	a, err := agents.Initialize(
		llm,
		[]tools.Tool{searchTool, calculator},
		agents.ZeroShotReactDescription,
	)
	require.NoError(t, err)

	result, err := chains.Run(context.Background(), a, "If a person lived three times as long as Jacklyn Zeman, how long would they live") //nolint:lll
	require.NoError(t, err)

	require.True(t, strings.Contains(result, "210"), "correct answer 210 not in response")
}
