use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use aleph_alpha_tokenizer::AlephAlphaTokenizer;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::tokenizer::Model;

static TEXT_LIST: &[&str] = &[
	"Ich esse Steak.",
	"Der Hund spielt im Garten.",
	"Gibt es genügend Impfstoff gegen FSME angesichts der steigenden Infektionszahlen?",
	"Ein Junge im Kindergarten spielt mit dem Ball.",
	"Wie definiert die Bundesregierung Clans und Clankriminalität?",
	"Gibt es genügend Impfstoff gegen Corona angesichts der steigenden Infektionszahlen?",
	"Steht vor dem Hintergrund der gestiegenen Infektionen ausreichend Impfstoff gegen FSME zur Verfügung?",
	"Wie viele Menschen starben durch die Folgen der Borreliose-Erkrankung?",
	"Liegen der Bundesregierung statistische Daten zu Todesfällen in Folge von Borreliose vor und wenn ja, wie lauten diese?",
	"Welche Abkommen mit auswärtigen Staaten bestehen seitens welcher Länder aktuell?",
	"Welche Vereinbarungen auf Landesebene bestehen mit Drittstaaten?",
];

fn compare_aleph_wordpiece(c: &mut Criterion) {
    let wordpiece = WordPiece::from_files("vocab.txt")
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let aleph_alpha = AlephAlphaTokenizer::from_vocab("vocab.txt").unwrap();
    let mut group = c.benchmark_group("Tokenizer");
    for (i, text) in TEXT_LIST.iter().cloned().enumerate() {
        let mut words = Vec::new();
        let mut o = 0;
        for word in text.split(' ') {
            words.push((word.to_string(), (o, o + word.len())));
            o += word.len() + 1;
        }
        group.bench_with_input(BenchmarkId::new("wordpiece", i), &i, |b, _| {
            b.iter(|| wordpiece.tokenize(black_box(words.clone())))
        });
        group.bench_with_input(BenchmarkId::new("aleph_alpha_model", i), &i, |b, _| {
            b.iter(|| aleph_alpha.tokenize(black_box(words.clone())))
        });
        group.bench_with_input(BenchmarkId::new("aleph_alpha", i), &i, |b, _| {
            let mut ids: Vec<u64> = Vec::new();
            let mut ranges = Vec::new();
            b.iter(|| {
                aleph_alpha.tokens_into(black_box(text), &mut ids, &mut ranges, None);
                black_box(&ids);
                black_box(&ranges);
            })
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = compare_aleph_wordpiece
}

criterion_main!(benches);
