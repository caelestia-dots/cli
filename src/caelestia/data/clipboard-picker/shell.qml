pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Qt5Compat.GraphicalEffects
import Quickshell
import Quickshell.Io
import Quickshell.Wayland

// Standalone clipboard picker used by the caelestia clipboard command.
ShellRoot {
    id: root

    readonly property string manifestPath: Quickshell.env("CAELESTIA_CLIPBOARD_MANIFEST") || ""
    readonly property string resultPath: Quickshell.env("CAELESTIA_CLIPBOARD_RESULT") || ""
    readonly property string refreshPath: Quickshell.env("CAELESTIA_CLIPBOARD_REFRESH") || ""
    readonly property string homePath: Quickshell.env("HOME") || ""
    readonly property string schemePath: Quickshell.env("CAELESTIA_CLIPBOARD_SCHEME") || `${root.homePath}/.local/state/caelestia/scheme.json`
    readonly property int textRowHeight: 48
    readonly property int imageRowHeight: textRowHeight * 6
    readonly property int popupWidth: 760
    readonly property int popupMaxHeight: 980

    property string query: ""
    property int currentIndex: 0
    property bool closing: false
    property bool actionInFlight: false
    property string pendingAction: ""
    property string animatingId: ""
    property var listViewObject: null

    // Keep this standalone picker in sync with the active Caelestia Material 3
    // scheme without importing the main shell (which would make it expensive
    // and couple its lifecycle to the picker).
    property color accentColor: "#8c4e43"
    property color accentTextColor: "#fff7f6"
    property color surfaceColor: "#fff8f6"
    property color surfaceContainerColor: "#fee9e6"
    property color surfaceContainerHighColor: "#fae3df"
    property color surfaceTextColor: "#3e2f2c"
    property color surfaceVariantTextColor: "#6d5b58"
    property color outlineColor: "#8a7673"
    property color outlineVariantColor: "#c3ada9"
    property color errorColor: "#a8364b"
    property color errorContainerColor: "#f97386"
    property color errorContainerTextColor: "#6e0523"

    function loadScheme(data) {
        if (!data)
            return;

        try {
            const colours = JSON.parse(data).colours || {};
            if (colours.primary)
                accentColor = `#${colours.primary}`;
            if (colours.onPrimary)
                accentTextColor = `#${colours.onPrimary}`;
            if (colours.surface)
                surfaceColor = `#${colours.surface}`;
            if (colours.surfaceContainer)
                surfaceContainerColor = `#${colours.surfaceContainer}`;
            if (colours.surfaceContainerHigh)
                surfaceContainerHighColor = `#${colours.surfaceContainerHigh}`;
            if (colours.onSurface)
                surfaceTextColor = `#${colours.onSurface}`;
            if (colours.onSurfaceVariant)
                surfaceVariantTextColor = `#${colours.onSurfaceVariant}`;
            if (colours.outline)
                outlineColor = `#${colours.outline}`;
            if (colours.outlineVariant)
                outlineVariantColor = `#${colours.outlineVariant}`;
            if (colours.error)
                errorColor = `#${colours.error}`;
            if (colours.errorContainer)
                errorContainerColor = `#${colours.errorContainer}`;
            if (colours.onErrorContainer)
                errorContainerTextColor = `#${colours.onErrorContainer}`;
        } catch (error) {
            // Keep the warm fallback palette if the scheme file is mid-write.
        }
    }

    // Keep the complete history as a plain JS array. Only the on-screen
    // filtered model needs ListView's delegate bookkeeping and transitions;
    // maintaining a second ListModel here doubled the work on every pin.
    property var allEntries: []

    ListModel {
        id: visibleEntries
    }

    function parseManifest(data) {
        const parsed = [];
        const lines = data.replace(/\r/g, "").split("\n");
        for (const line of lines) {
            if (!line)
                continue;

            const firstTab = line.indexOf("\t");
            if (firstTab < 1)
                continue;

            const id = line.slice(0, firstTab);
            const remainder = line.slice(firstTab + 1);
            const secondTab = remainder.indexOf("\t");
            if (secondTab < 1)
                continue;

            const kind = remainder.slice(0, secondTab);
            const payload = remainder.slice(secondTab + 1);

            if (kind === "image") {
                const pathEnd = payload.indexOf("\t");
                const pinEnd = payload.lastIndexOf("\t");
                const source = pathEnd >= 0 ? payload.slice(0, pathEnd) : payload;
                const searchText = pathEnd >= 0
                    ? payload.slice(pathEnd + 1, pinEnd > pathEnd ? pinEnd : payload.length)
                    : "image";
                const pinned = pinEnd > pathEnd && payload.slice(pinEnd + 1) === "1";
                parsed.push({
                    entryId: id,
                    kind: kind,
                    displayText: "",
                    searchText: searchText,
                    source: source,
                    pinned: pinned
                });
            } else {
                const pinEnd = payload.lastIndexOf("\t");
                parsed.push({
                    entryId: id,
                    kind: "text",
                    displayText: pinEnd >= 0 ? payload.slice(0, pinEnd) : payload,
                    searchText: pinEnd >= 0 ? payload.slice(0, pinEnd) : payload,
                    source: "",
                    pinned: pinEnd >= 0 && payload.slice(pinEnd + 1) === "1"
                });
            }
        }

        return parsed;
    }

    function setModelItem(model, index, item) {
        // Set roles explicitly instead of passing an object from another
        // ListModel. This keeps the existing delegate alive while its data
        // changes, which is important for smooth move/remove transitions.
        model.setProperty(index, "entryId", item.entryId);
        model.setProperty(index, "kind", item.kind);
        model.setProperty(index, "displayText", item.displayText);
        model.setProperty(index, "searchText", item.searchText);
        model.setProperty(index, "source", item.source);
        model.setProperty(index, "pinned", item.pinned);
    }

    function reconcileModel(model, desired) {
        const wanted = {};
        for (const item of desired)
            wanted[item.entryId] = true;

        // Remove stale rows first so a deleted row animates out from its
        // current position instead of being moved to the end before removal.
        for (let i = model.count - 1; i >= 0; --i) {
            if (!wanted[model.get(i).entryId])
                model.remove(i, 1);
        }

        for (let target = 0; target < desired.length; ++target) {
            const item = desired[target];
            let existing = -1;
            for (let i = target; i < model.count; ++i) {
                if (model.get(i).entryId === item.entryId) {
                    existing = i;
                    break;
                }
            }

            if (existing < 0) {
                model.insert(target, item);
            } else {
                if (existing !== target)
                    model.move(existing, target, 1);
                setModelItem(model, target, item);
            }
        }
    }

    function loadManifest(data) {
        const parsed = parseManifest(data);
        allEntries = parsed;
        refilter();
    }

    function refilter() {
        const needle = query.trim().toLocaleLowerCase();
        const previousIndex = currentIndex;
        const currentId = visibleEntries.count > currentIndex
            ? visibleEntries.get(currentIndex).entryId
            : "";
        const filtered = [];

        for (let i = 0; i < allEntries.length; ++i) {
            const item = allEntries[i];
            if (!needle || item.searchText.toLocaleLowerCase().includes(needle))
                filtered.push(item);
        }

        reconcileModel(visibleEntries, filtered);

        if (currentId) {
            let currentPosition = -1;
            for (let i = 0; i < visibleEntries.count; ++i) {
                if (visibleEntries.get(i).entryId === currentId) {
                    currentPosition = i;
                    break;
                }
            }
            // If the selected row was deleted, keep the focus at the same
            // position so the next row takes over instead of jumping to top.
            currentIndex = currentPosition >= 0
                ? currentPosition
                : (visibleEntries.count ? Math.min(previousIndex, visibleEntries.count - 1) : 0);
        } else {
            currentIndex = visibleEntries.count ? Math.min(previousIndex, visibleEntries.count - 1) : 0;
        }

        if (listViewObject && listViewObject.count > 0)
            listViewObject.positionViewAtIndex(currentIndex, ListView.Contain);

        if (!closing)
            actionInFlight = false;
    }

    function moveSelection(delta) {
        if (!visibleEntries.count)
            return;

        currentIndex = Math.max(0, Math.min(visibleEntries.count - 1, currentIndex + delta));
        if (listViewObject)
            listViewObject.positionViewAtIndex(currentIndex, ListView.Contain);
    }

    function requestAction(action, id) {
        if (closing || actionInFlight)
            return;

        actionInFlight = true;
        pendingAction = action;
        if (action === "pin" || action === "delete") {
            animatingId = id || "";
            animationReset.restart();
        }
        closing = action === "paste" || action === "close";
        if (resultPath) {
            writeResult.command = [
                "sh", "-c", "printf '%s\\t%s' \"$1\" \"$2\" > \"$3\"",
                "clipboard-picker",
                action,
                id || "",
                resultPath
            ];
            writeResult.running = true;
        } else {
            Qt.quit();
        }
    }

    function chooseCurrent() {
        if (visibleEntries.count > 0)
            requestAction("paste", visibleEntries.get(currentIndex).entryId);
    }

    function togglePin(id) {
        requestAction("pin", id);
    }

    function deleteEntry(id) {
        requestAction("delete", id);
    }

    Timer {
        id: animationReset
        interval: 380
        repeat: false
        onTriggered: root.animatingId = ""
    }

    FileView {
        id: manifestReader

        path: root.manifestPath
        preload: true
        blockLoading: true
        watchChanges: true
        Component.onCompleted: {
            root.loadManifest(text());
        }
        onLoaded: root.loadManifest(text())
        onFileChanged: reload()
    }

    // The shell script replaces the manifest atomically, then updates this
    // marker. Watching the marker avoids parsing an incomplete file and lets
    // ListView animate the precise rows that moved or disappeared.
    FileView {
        id: refreshReader

        path: root.refreshPath
        preload: true
        blockLoading: true
        watchChanges: true
        onFileChanged: manifestReader.reload()
    }

    FileView {
        id: schemeReader

        path: root.schemePath
        preload: true
        blockLoading: true
        watchChanges: true
        Component.onCompleted: {
            root.loadScheme(text());
        }
        onLoaded: root.loadScheme(text())
        onFileChanged: reload()
    }

    Process {
        id: writeResult

        onExited: {
            const action = root.pendingAction;
            root.pendingAction = "";
            if (action === "paste" || action === "close" || !root.resultPath)
                Qt.quit();
        }
    }

    Variants {
        model: Quickshell.screens

        PanelWindow {
            id: pickerWindow

            required property ShellScreen modelData
            screen: modelData

            visible: true
            color: "transparent"
            surfaceFormat.opaque: false
            focusable: true

            WlrLayershell.namespace: "caelestia-clipboard-picker"
            WlrLayershell.layer: WlrLayer.Overlay
            WlrLayershell.exclusionMode: ExclusionMode.Ignore
            WlrLayershell.keyboardFocus: WlrKeyboardFocus.Exclusive

            anchors.top: true
            anchors.bottom: true
            anchors.left: true
            anchors.right: true

            Item {
                id: overlay
                anchors.fill: parent
                focus: true

                // The panel covers the whole screen so it can receive input;
                // this background hit target closes the picker when the click
                // lands outside the centered popup. Its negative z keeps all
                // popup controls above it.
                MouseArea {
                    id: outsideClick
                    anchors.fill: parent
                    z: -1
                    acceptedButtons: Qt.LeftButton | Qt.RightButton
                    onClicked: root.requestAction("close", "")
                }

            Rectangle {
                id: popup

                anchors.centerIn: parent
                width: root.popupWidth
                height: Math.min(root.popupMaxHeight, header.height + listView.contentHeight + 32)
                radius: 20
                color: root.surfaceColor
                border.width: 2
                border.color: Qt.alpha(root.accentColor, 0.5)
                clip: true

                scale: root.closing ? 0.96 : 1
                opacity: root.closing ? 0 : 1

                Behavior on scale {
                    NumberAnimation { duration: 120; easing.type: Easing.OutCubic }
                }
                Behavior on opacity {
                    NumberAnimation { duration: 120; easing.type: Easing.OutCubic }
                }

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 10

                    RowLayout {
                        id: header
                        Layout.fillWidth: true
                        spacing: 10

                        Text {
                            text: ">"
                            color: root.accentColor
                            font.family: "JetBrains Mono NF"
                            font.pixelSize: 22
                            font.bold: true
                        }

                        TextField {
                            id: searchInput
                            Layout.fillWidth: true
                            Layout.preferredHeight: root.textRowHeight
                            color: root.surfaceTextColor
                            selectionColor: Qt.alpha(root.accentColor, 0.4)
                            selectedTextColor: root.surfaceTextColor
                            font.family: "JetBrains Mono NF"
                            font.pixelSize: 17
                            clip: true
                            focus: true
                            selectByMouse: true
                            verticalAlignment: Text.AlignVCenter
                            placeholderText: "搜索剪贴板"
                            placeholderTextColor: root.outlineColor
                            background: Item {}

                            onTextChanged: {
                                root.query = text;
                                root.currentIndex = 0;
                                root.refilter();
                            }

                            Keys.onPressed: event => {
                                if (event.key === Qt.Key_Down) {
                                    root.moveSelection(1);
                                    event.accepted = true;
                                } else if (event.key === Qt.Key_Up) {
                                    root.moveSelection(-1);
                                    event.accepted = true;
                                } else if (event.key === Qt.Key_PageDown) {
                                    root.moveSelection(5);
                                    event.accepted = true;
                                } else if (event.key === Qt.Key_PageUp) {
                                    root.moveSelection(-5);
                                    event.accepted = true;
                                } else if (event.key === Qt.Key_Return || event.key === Qt.Key_Enter) {
                                    root.chooseCurrent();
                                    event.accepted = true;
                                } else if (event.key === Qt.Key_Escape) {
                                    root.requestAction("close", "");
                                    event.accepted = true;
                                }
                            }
                        }

                        Text {
                            text: `${visibleEntries.count}`
                            color: root.surfaceVariantTextColor
                            font.family: "JetBrains Mono NF"
                            font.pixelSize: 14
                            Layout.alignment: Qt.AlignVCenter
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 1
                        color: Qt.alpha(root.outlineVariantColor, 0.35)
                    }

                    ListView {
                        id: listView

                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 8
                        model: visibleEntries
                        currentIndex: root.currentIndex
                        boundsBehavior: Flickable.StopAtBounds
                        ScrollBar.vertical: ScrollBar { }

                        add: Transition {
                            NumberAnimation {
                                property: "opacity"
                                from: 0
                                to: 1
                                duration: 120
                                easing.type: Easing.OutCubic
                            }
                        }

                        remove: Transition {
                            ParallelAnimation {
                                NumberAnimation {
                                    property: "opacity"
                                    to: 0
                                    duration: 130
                                    easing.type: Easing.InCubic
                                }
                                NumberAnimation {
                                    property: "scale"
                                    to: 0.96
                                    duration: 130
                                    easing.type: Easing.InCubic
                                }
                            }
                        }

                        move: Transition {
                            NumberAnimation {
                                property: "y"
                                duration: 190
                                easing.type: Easing.OutCubic
                            }
                        }

                        moveDisplaced: Transition {
                            NumberAnimation {
                                property: "y"
                                duration: 190
                                easing.type: Easing.OutCubic
                            }
                        }

                        addDisplaced: Transition {
                            NumberAnimation {
                                property: "y"
                                duration: 160
                                easing.type: Easing.OutCubic
                            }
                        }

                        removeDisplaced: Transition {
                            NumberAnimation {
                                property: "y"
                                duration: 170
                                easing.type: Easing.OutCubic
                            }
                        }

                        Component.onCompleted: root.listViewObject = listView

                        delegate: Rectangle {
                            id: delegateRoot

                            required property int index
                            required property string entryId
                            required property string kind
                            required property string displayText
                            required property string source
                            required property bool pinned

                            width: listView.width - (listView.ScrollBar.vertical.visible ? 12 : 0)
                            height: kind === "image" ? root.imageRowHeight : root.textRowHeight
                            z: delegateRoot.entryId === root.animatingId ? 5 : 0
                            radius: kind === "image" ? 14 : 12
                            // Cache the expensive image/mask composition while
                            // it is translated by ListView's move transition.
                            layer.enabled: kind === "image"
                            color: index === root.currentIndex ? root.surfaceContainerHighColor : root.surfaceContainerColor
                            border.width: index === root.currentIndex ? 2 : 0
                            border.color: root.accentColor
                            clip: true

                            Item {
                                anchors.fill: parent
                                anchors.margins: 3
                                visible: delegateRoot.kind === "image"

                                Image {
                                    id: imageSource

                                    anchors.fill: parent
                                    visible: false
                                    source: delegateRoot.source ? `file://${delegateRoot.source}` : ""
                                    // Decode previews at their display size; the
                                    // clipboard cache can contain multi-megapixel
                                    // screenshots that do not need full-res
                                    // textures inside this picker.
                                    sourceSize: Qt.size(Math.max(1, delegateRoot.width), Math.max(1, delegateRoot.height))
                                    fillMode: Image.PreserveAspectCrop
                                    asynchronous: true
                                    cache: true
                                    mipmap: true
                                }

                                Rectangle {
                                    id: imageMask

                                    anchors.fill: parent
                                    visible: false
                                    color: "white"
                                    radius: Math.max(0, delegateRoot.radius - 3)
                                }

                                OpacityMask {
                                    anchors.fill: imageSource
                                    source: imageSource
                                    maskSource: imageMask
                                }
                            }

                            Text {
                                anchors.fill: parent
                                anchors.leftMargin: 16
                                anchors.rightMargin: 16
                                visible: delegateRoot.kind !== "image"
                                text: delegateRoot.displayText
                                color: root.surfaceTextColor
                                font.family: "JetBrains Mono NF"
                                font.pixelSize: 16
                                elide: Text.ElideRight
                                verticalAlignment: Text.AlignVCenter
                                maximumLineCount: 1
                            }

                            Row {
                                anchors.top: parent.top
                                anchors.right: parent.right
                                anchors.topMargin: 8
                                anchors.rightMargin: 8
                                spacing: 5
                                z: 3

                                Rectangle {
                                    width: pinLabel.implicitWidth + 18
                                    height: 30
                                    radius: 10
                                    color: delegateRoot.pinned ? Qt.alpha(root.accentColor, 0.9) : Qt.alpha(root.surfaceColor, 0.9)
                                    border.width: 1
                                    border.color: delegateRoot.pinned ? root.accentColor : Qt.alpha(root.accentColor, 0.4)

                                    Text {
                                        id: pinLabel
                                        anchors.centerIn: parent
                                        text: delegateRoot.pinned ? "取消置顶" : "置顶"
                                        color: delegateRoot.pinned ? root.accentTextColor : root.accentColor
                                        font.family: "Noto Sans CJK SC"
                                        font.pixelSize: 13
                                    }

                                    MouseArea {
                                        anchors.fill: parent
                                        onClicked: {
                                            root.currentIndex = delegateRoot.index;
                                            root.togglePin(delegateRoot.entryId);
                                        }
                                    }
                                }

                                Rectangle {
                                    width: deleteLabel.implicitWidth + 18
                                    height: 30
                                    radius: 10
                                    color: Qt.alpha(root.errorContainerColor, 0.25)
                                    border.width: 1
                                    border.color: Qt.alpha(root.errorColor, 0.7)

                                    Text {
                                        id: deleteLabel
                                        anchors.centerIn: parent
                                        text: "删除"
                                        color: root.errorContainerTextColor
                                        font.family: "Noto Sans CJK SC"
                                        font.pixelSize: 13
                                    }

                                    MouseArea {
                                        anchors.fill: parent
                                        onClicked: {
                                            root.currentIndex = delegateRoot.index;
                                            root.deleteEntry(delegateRoot.entryId);
                                        }
                                    }
                                }
                            }

                            MouseArea {
                                anchors.fill: parent
                                hoverEnabled: true
                                onEntered: root.currentIndex = delegateRoot.index
                                onClicked: root.requestAction("paste", delegateRoot.entryId)
                            }
                        }
                    }

                    Text {
                        visible: visibleEntries.count === 0
                        Layout.fillWidth: true
                        Layout.preferredHeight: root.textRowHeight
                        text: "没有匹配的剪贴板内容"
                        color: root.surfaceVariantTextColor
                        font.family: "JetBrains Mono NF"
                        font.pixelSize: 16
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                }
            }
            }

            Component.onCompleted: {
                searchInput.forceActiveFocus();
            }
        }
    }
}
