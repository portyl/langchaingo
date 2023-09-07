package documentloaders

import (
	"context"

	"github.com/portyl/langchaingo/schema"
	"github.com/portyl/langchaingo/textsplitter"
)

// Loader is the interface for loading and splitting documents from a source.
type Loader interface {
	// Loads loads from a source and returns documents.
	Load(context.Context) ([]schema.Document, error)
	// LoadAndSplit loads from a source and splits the documents using a text splitter.
	LoadAndSplit(context.Context, textsplitter.TextSplitter) ([]schema.Document, error)
}
