import * as fs from 'node:fs/promises';

const custom = [
    "¿? question upside down reversed spanish",
    "← left arrow",
    "↑ up arrow",
    "→ right arrow",
    "↓ down arrow",
    "←↑→↓ all directions up down left right arrows",
    "⇇ leftwards paired arrows",
    "⇉ rightwards paired arrows",
    "⇈ upwards paired arrows",
    "⇊ downwards paired arrows",
    "⬱ three leftwards arrows",
    "⇶ three rightwards arrows",
    "• dot circle separator",
    "「」 japanese quote square bracket",
    "¯\_(ツ)_/¯ shrug idk i dont know",
    "(ง🔥ﾛ🔥)ง person with fire eyes eyes on fire",
    "↵ enter key return",

    "— em dash",
    "󰖳 windows super key",
]

interface EmojiItem {
    group?: number,
    hexcode: string,
    label: string,
    order?: number,
    tags?: string[],
    unicode: string,
    emoticon?: string | string[],
}

interface NFList {
    [key: string]: {
        char: string,
        code: string,
    }
}

async function fetchEmojis() {
    const link = "https://raw.githubusercontent.com/milesj/emojibase/refs/tags/emojibase-data%4016.0.3/packages/data/en/compact.raw.json"
    const res = await fetch(link)
    const data = await res.json()
    return data as EmojiItem[]
}

async function fetchNF() {
    const link = "https://raw.githubusercontent.com/ryanoasis/nerd-fonts/refs/tags/v3.4.0/glyphnames.json"
    const res = await fetch(link)
    const data = await res.json()
    delete data["METADATA"]
    return data as NFList
}

function formatEmojis(items: EmojiItem[]) {
    return items.map((item) => {
        const line = [item.unicode]

        if (item.emoticon) {
            const emoticon = typeof item.emoticon === "object" ? item.emoticon : [item.emoticon]
            line.push(...emoticon)
        }

        line.push(item.label)

        if (item.tags) {
            line.push(...item.tags)
        }

        return line.join(' ')
    })
}

function formatNF(list: NFList) {
    const buckets = {} as { [key: string]: string[] }
    for (const [key, { char }] of Object.entries(list)) {
        (buckets[char] ??= []).push(`nf-${key}`) // add prefix!
    }

    const result = Object.entries(buckets).map(([char, keys]) => `${char} ${keys.join(" ")}`)
    return result
}

const emojis = formatEmojis(await fetchEmojis())
const nf = formatNF(await fetchNF())

const list = [
    ...custom,
    ...emojis,
    ...nf,
]

async function writeToFile(list: string[]) {
    const txt = list.join("\n")
    await fs.writeFile("./emojis.txt", txt)
}

await writeToFile(list)
